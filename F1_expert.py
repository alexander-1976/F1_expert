import os
import json
import re
import time
import random
from datetime import datetime
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import Dataset, DatasetDict
from evaluate import load
from sentence_transformers import SentenceTransformer, util
import faiss
import mwparserfromhell
import wikipediaapi

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ========== PATHS ==========
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = "processed"
DATASET_PATH = os.path.join(PROCESSED_DIR, "f1_qa_dataset.jsonl")
MODEL_OUTPUT_DIR = os.path.join("models", "rubert_tiny2_qa")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ========== UTILS ==========
def safe_filename(name):
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def normalize_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', '', text)).strip()

def is_answer_in_context(context, answer):
    return normalize_text(answer) in normalize_text(context)

def remove_duplicates(data):
    seen = set()
    unique_data = []
    for item in data:
        key = (normalize_text(item["question"]), normalize_text(item["context"]))
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    print(f"Удалено дубликатов: {len(data) - len(unique_data)}")
    return unique_data

# ========== WIKI SCRAPER ==========
class F1WikiRawSaver:
    def __init__(self, lang='ru', storage_path='data', max_workers=4):
        self.wiki = wikipediaapi.Wikipedia(
            language=lang,
            user_agent="F1WikiRawSaver/1.1 (research@example.com)",
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        self.storage_path = os.path.abspath(storage_path)
        self.max_workers = max_workers
        self.visited_articles = set()
        self.visited_categories = set()
        self.failed_articles = []
        self.categories = [
            "Категория:Гран-при Формулы-1",
            "Категория:Этапы Формулы-1",
            "Категория:Сезоны Формулы-1",
            "Категория:Чемпионаты Формулы-1",
            "Категория:Конструкторы Формулы-1",
            "Категория:Команды Формулы-1",
            "Категория:Гонщики Формулы-1",
            "Категория:Пилоты Формулы-1",
            "Категория:Двигатели Формулы-1",
            "Категория:Моторы Формулы-1",
            "Категория:Шины Формулы-1",
            "Категория:Покрышки Формулы-1",
            "Категория:Автодромы Формулы-1",
            "Категория:Трассы Формулы-1",
            "Категория:Аварии в Формуле-1",
            "Категория:Происшествия в Формуле-1",
            "Категория:История Формулы-1",
            "Категория:Инженеры Формулы-1",
            "Категория:Техники Формулы-1",
            "Категория:Руководители Формулы-1",
            "Категория:Владельцы команд Формулы-1"
        ]
        self._setup_directory_structure()

    def _setup_directory_structure(self):
        os.makedirs(os.path.join(self.storage_path, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, 'logs'), exist_ok=True)

    def _get_all_titles(self, category, depth=2):
        titles = set()
        try:
            category_page = self.wiki.page(category)
            if not category_page.exists():
                alt_category = category.replace("Формулы-1", "Формулы 1")
                category_page = self.wiki.page(alt_category)
                if not category_page.exists():
                    logging.warning(f"Категория не найдена: {category}")
                    return titles
            for title, member in category_page.categorymembers.items():
                if member.ns == wikipediaapi.Namespace.CATEGORY and depth > 0:
                    if title in self.visited_categories:
                        continue
                    self.visited_categories.add(title)
                    titles.update(self._get_all_titles(title, depth-1))
                elif member.ns == wikipediaapi.Namespace.MAIN:
                    if (title, category) in self.visited_articles:
                        continue
                    self.visited_articles.add((title, category))
                    titles.add((title, category))
                time.sleep(0.1)
        except Exception as e:
            logging.warning(f"Ошибка при обработке категории {category}: {str(e)}")
        return titles

    def _save_article(self, title, text):
        filename = safe_filename(title) + ".txt"
        filepath = os.path.join(self.storage_path, 'raw', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

    def _process_article(self, title, category):
        base_delay = 1.0
        for attempt in range(5):
            try:
                page = self.wiki.page(title)
                if not page.exists() or not page.text.strip():
                    self.failed_articles.append({
                        'title': title,
                        'category': category,
                        'reason': 'not_found_or_empty',
                        'timestamp': datetime.now().isoformat()
                    })
                    return
                try:
                    parsed = mwparserfromhell.parse(page.text)
                    clean_text = parsed.strip_code()
                except Exception:
                    clean_text = page.text
                self._save_article(title, clean_text)
                return
            except Exception as e:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                logging.warning(f"Попытка {attempt+1} для '{title}' ({category}): {str(e)}. Жду {delay:.1f} сек.")
                time.sleep(delay)
        self.failed_articles.append({
            'title': title,
            'category': category,
            'reason': 'max_attempts_failed',
            'timestamp': datetime.now().isoformat()
        })

    def collect_and_save_all(self):
        all_titles = set()
        for category in self.categories:
            logging.info(f"Сбор: {category}")
            all_titles.update(self._get_all_titles(category))
        logging.info(f"Всего уникальных пар (статья, категория): {len(all_titles)}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for title, category in all_titles:
                futures.append(executor.submit(self._process_article, title, category))
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Сохранение статей"):
                pass
        if self.failed_articles:
            failed_path = os.path.join(self.storage_path, 'logs', 'failed_articles.json')
            with open(failed_path, 'w', encoding='utf-8') as f:
                json.dump(self.failed_articles, f, ensure_ascii=False, indent=2)
            logging.info(f"Ошибок при обработке статей: {len(self.failed_articles)} (см. {failed_path})")
        progress_path = os.path.join(self.storage_path, 'logs', 'progress.json')
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump({
                'visited_articles': [list(x) for x in self.visited_articles],
                'visited_categories': list(self.visited_categories),
                'total': len(self.visited_articles),
                'last_saved': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("Запуск парсинга статей...")
    wiki_saver = F1WikiRawSaver(storage_path='data', max_workers=4)
    wiki_saver.collect_and_save_all()
    print("Сбор статей завершён!")

# ========== ГЕНЕРАЦИЯ QA-ПАР ==========
def build_corpus(raw_dir):
    data = []
    for fname in tqdm(os.listdir(raw_dir), desc="Чтение файлов"):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(raw_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            title = fname[:-4]
            data.append({
                'title': title,
                'text': text,
                'length': len(text),
                'file': fname
            })
        except Exception as e:
            print(f"Ошибка при чтении {fname}: {e}")
    return data

def generate_qa_pairs(corpus):
    model_name = "cointegrated/rut5-base-multitask"
    try:
        print("Загрузка генеративной модели...")
        gen_tokenizer = T5Tokenizer.from_pretrained(model_name)
        gen_model = T5ForConditionalGeneration.from_pretrained(model_name).to("cpu")
        print("Модель загружена!")
    except Exception as e:
        print(f"Ошибка загрузки генеративной модели: {e}")
        raise

    qa_data = []
    for item in tqdm(corpus, desc="Генерация QA-предложений"):
        context = item['text']
        inputs = gen_tokenizer(f"ask | {context[:1024]}", return_tensors="pt").to(gen_model.device)
        question = gen_model.generate(**inputs, num_beams=2, max_length=64, early_stopping=True)
        question = gen_tokenizer.decode(question[0], skip_special_tokens=True)
        inputs = gen_tokenizer(f"comprehend | {context[:1024]}. Вопрос: {question}?", return_tensors="pt").to(gen_model.device)
        answer = gen_model.generate(**inputs, num_beams=2, max_length=64, early_stopping=True)
        answer = gen_tokenizer.decode(answer[0], skip_special_tokens=True)
        if not answer or answer.lower() not in context.lower():
            continue
        sample = {
            "context": context,
            "question": question,
            "answer": answer
        }
        qa_data.append(sample)

    dataset_path = os.path.join(PROCESSED_DIR, "f1_qa_dataset.jsonl")
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for item in qa_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"QA-датасет создан: {dataset_path}")
    return qa_data

# ========== ЗАГРУЗКА ДАННЫХ ==========
print("Формирование корпуса...")
corpus = build_corpus(RAW_DIR)
print(f"Всего статей: {len(corpus)}")

print("Генерация QA-пар...")
qa_data = generate_qa_pairs(corpus)

print("Фильтрация дубликатов...")
unique_data = remove_duplicates(qa_data)
dataset = Dataset.from_list(unique_data).train_test_split(test_size=0.1, shuffle=True, seed=42)
print(f"Датасет загружен: {len(dataset['train'])} тренировочных записей, {len(dataset['test'])} тестовых")

# ========== ТОКЕНИЗАЦИЯ ==========
def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sample_idx = sample_mapping[i]
        answer = examples["answer"][sample_idx]
        context = examples["context"][sample_idx]

        answer_start = context.find(answer)
        if answer_start == -1:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            continue

        answer_end = answer_start + len(answer) - 1
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        token_start_index = 0
        while sequence_ids[token_start_index] != context_index:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_index:
            token_end_index -= 1

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start:
            token_start_index += 1
        tokenized_examples["start_positions"].append(token_start_index - 1)

        while offsets[token_end_index][1] >= answer_end:
            token_end_index -= 1
        tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

# ========== ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА ==========
model_name = "cointegrated/rubert-tiny2"
try:
    print("Загрузка модели и токенизатора...")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name).to("cpu")
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    raise

# ========== ТОКЕНИЗАЦИЯ ==========
print("Токенизация данных...")
tokenized_datasets = dataset.map(
    lambda x: prepare_train_features(x, tokenizer),
    batched=True,
    remove_columns=["context", "question", "answer"]
)

# ========== ОЦЕНКА КАЧЕСТВА ==========
try:
    print("Загрузка метрики SQuAD...")
    metric = load("squad")
except Exception as e:
    print(f"Ошибка загрузки метрики: {e}")
    raise

def compute_metrics(p, dataset, tokenizer):
    predictions = []
    references = []
    for i in range(len(p.predictions)):
        context = dataset[i]["context"]
        question = dataset[i]["question"]
        true_answer = dataset[i]["answer"]
        inputs = tokenizer(question, context, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)
        pred_answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index+1], skip_special_tokens=True)
        predictions.append({"prediction_text": pred_answer, "id": str(i)})
        references.append({
            "id": str(i),
            "answers": {
                "text": [true_answer],
                "answer_start": [context.find(true_answer)]
            }
        })
    return metric.compute(predictions=predictions, references=references)

# ========== ОБУЧЕНИЕ ==========
print("Настройка параметров обучения...")
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    learning_rate=1e-5,
    weight_decay=0.1,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=lambda p: compute_metrics(p, dataset["test"], tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Начало обучения...")
trainer.train()
print("Обучение завершено!")
trainer.save_model(MODEL_OUTPUT_DIR)
print(f"Модель сохранена: {MODEL_OUTPUT_DIR}")

# ========== СРАВНЕНИЕ КАЧЕСТВА ==========
print("Сравнение качества на train и test...")
train_metrics = compute_metrics(trainer.predict(tokenized_datasets["train"]), dataset["train"], tokenizer)
test_metrics = compute_metrics(trainer.predict(tokenized_datasets["test"]), dataset["test"], tokenizer)
print("Метрики на train:", {k: f"{v:.2f}" for k, v in train_metrics.items()})
print("Метрики на test:", {k: f"{v:.2f}" for k, v in test_metrics.items()})

def plot_comparison(train_metrics, test_metrics):
    metrics = ['exact_match', 'f1']
    x = np.arange(len(metrics))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.175, [train_metrics[m] for m in metrics], width=0.35, label='Train', color='skyblue')
    plt.bar(x + 0.175, [test_metrics[m] for m in metrics], width=0.35, label='Test', color='salmon')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'performance_comparison.png'))
    plt.show()

plot_comparison(train_metrics, test_metrics)

# ========== ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС ==========
# Создание эмбеддингов и FAISS-индекса
context_embeddings = None
faiss_index = None

def create_context_embeddings(dataset):
    embedder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    contexts = [normalize_text(item["context"]) for item in dataset]
    embeddings = embedder.encode(context_embeddings, convert_to_tensor=True)
    return embeddings

def create_faiss_index(context_embeddings):
    dimension = context_embeddings.shape[1]
    nlist = 100
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index.train(context_embeddings.numpy())
    index.add(context_embeddings.numpy())
    return index

def filter_context_by_keywords(question, context):
    question_words = set(normalize_text(question).split())
    context_words = set(normalize_text(context).split())
    return len(question_words & context_words) > 0

# ========== ПРЕДСКАЗАНИЕ ОТВЕТА ==========
def predict_answer(question):
    question = normalize_text(question)
    embedder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    question_embedding = embedder.encode([question])
    distances, indices = faiss_index.search(question_embedding, 10)
    for idx in indices[0]:
        index = int(idx)
        if 0 <= index < len(dataset["train"]):
            context = dataset["train"][index]["context"]
            if filter_context_by_keywords(question, context):
                break
    else:
        context = "Контекст не найден"
    inputs = tokenizer(question, context, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        start_index = torch.argmax(outputs.start_logits)
        end_index = torch.argmax(outputs.end_logits)
        answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index+1], skip_special_tokens=True)
    return answer, context

def collect_feedback():
    feedback_data = []
    print("Задайте вопросы для оценки (введите 'exit' для выхода):")
    while True:
        question = input("Вы: ").strip()
        if question.lower() == "exit":
            break
        answer, context = predict_answer(question)
        print(f"Ответ: {answer}")
        try:
            rating = int(input("Оцените ответ (1-5): "))
            if 1 <= rating <= 5:
                feedback_data.append({
                    "question": question,
                    "context": context,
                    "answer": answer,
                    "user_rating": rating
                })
            else:
                print("Неверная оценка. Пропущено.")
        except ValueError:
            print("Пожалуйста, введите число от 1 до 5.")
    return feedback_data

def retrain_model(feedback_data):
    if not feedback_data:
        print("Нет данных для дообучения")
        return
    low_rated = [item for item in feedback_data if item["user_rating"] <= 3]
    if not low_rated:
        print("Нет примеров с низкой оценкой. Дообучение не требуется")
        return

    new_dataset = Dataset.from_list([
        {"context": d["context"], "question": d["question"], "answer": d["answer"]}
        for d in low_rated
    ]).train_test_split(test_size=0.1, seed=42)

    global dataset
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset["train"] = dataset["train"].concatenate(new_dataset["train"])

    global context_embeddings, faiss_index
    context_embeddings = create_context_embeddings(dataset["train"])
    faiss_index = create_faiss_index(context_embeddings)

    print("Модель обновлена!")

# ========== ОСНОВНОЙ ЦИКЛ ==========
print("Создание эмбеддингов и индекса FAISS...")
context_embeddings = create_context_embeddings(dataset["train"])
faiss_index = create_faiss_index(context_embeddings)

if __name__ == "__main__":
    print("Запуск проекта...")
    feedback_data = collect_feedback()
    if feedback_data:
        retrain_model(feedback_data)
    print("Модель готова к использованию!")