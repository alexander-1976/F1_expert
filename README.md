# F1_Expert: Вопросно-ответная модель по Формуле-1 на основе Википедии

## Краткое описание проекта

**F1_Expert** — это end-to-end проект по созданию полноценной **вопросно-ответной системы (QA system)** по тематике **Формулы-1** на русском языке.  
Он охватывает весь pipeline NLP-модели — от сбора данных до обучения и взаимодействия с пользователем.  

Проект выполнен **полностью под CPU**, в связи с отсутствием доступа к производительному GPU. Это повлияло на выбор архитектур: использовались **легковесные модели**, оптимизированные для inference без ускорителей.


## Цель проекта

- Построить интерактивную QA-систему на русском языке
- Использовать открытые источники (Википедия)
- Реализовать устойчивую работу на **средних компьютерах без GPU**
- Поддержать легкое **дообучение** на пользовательской обратной связи


## Функционал кода
- Скачивает статьи из Википедии по категориям, связанным с Формулой-1
- Сохраняет их в структурированном виде
- Формирует корпус текстов и генерирует датасет вопрос-ответ
- Обучает модель `rubert-tiny2` на датасете вопрос-ответ
- Использует метрику SQuAD для оценки качества
- Интерфейс в консоли: пользователь задаёт вопрос → получает ответ → ставит оценку (1–5)  
- Модель дообучается на низкооцененных примерах
- Использует эмбеддинги и FAISS


## Структура проекта

```
F1_expert/
├── F1_expert.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/
├── processed/
│   └── f1_qa_dataset.jsonl
├── models/
│   └── rubert_tiny2_qa/
└── logs/
```
## Использованные модели

| Назначение               | Модель                                                        | Особенности                                 |
|--------------------------|----------------------------------------------------------------|----------------------------------------------|
| Генерация QA             | `cointegrated/rut5-base-multitask`                            | Модель T5, fine-tuned для QA генерации       |
| Извлечение ответа        | `cointegrated/rubert-tiny2`                                   | Легковесный русский BERT                     |
| Семантический поиск      | `sentence-transformers/distiluse-base-multilingual-cased-v2` | Многоязычная модель Sentence Transformers     |
