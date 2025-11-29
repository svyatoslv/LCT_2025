# Решение хакатона ЛЦТ 2025

Оптимизация SQL запросов для Data Lakehouse (Trino + Iceberg + S3) с помощью мультиагентного пайплайна.

## Описание

От нас требовалось оптимизировать SQL запросы, полученные на вход. Мы решили данную задачу с помощью мультиагентного пайплайна на основе LangChain и LangGraph.

### Архитектура пайплайна

1. **Query Analyzer** - анализирует SQL запросы, выявляет паттерны использования
2. **Query Summarizer** - агрегирует результаты анализа
3. **DDL Optimizer** - генерирует оптимизированные DDL
4. **Migrations Creator** - создаёт миграции для применения изменений
5. **Query Optimizer** - переписывает запросы под новую структуру
6. **Critic & Judge** - проверяют корректность оптимизации

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/svyatoslv/LCT_2025.git
cd LCT_2025

# Установка зависимостей
pip install -r requirements.txt
```

## Запуск решения

### Использование Python скрипта

```bash
# Базовый запуск
python run_solution.py --input ./Датасет/flights.json --output ./output_result.json

# С указанием API ключа
python run_solution.py \
    --input ./Датасет/flights.json \
    --output ./output_result.json \
    --api-key YOUR_API_KEY

# Полный список параметров
python run_solution.py \
    --input ./Датасет/flights.json \
    --output ./output_result.json \
    --api-key YOUR_API_KEY \
    --api-base https://cloud.m1r0.ru/v1 \
    --model qwen3-coder:30b \
    --max-concurrent 2 \
    --retries 3
```

### Параметры запуска

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--input, -i` | Путь к входному JSON файлу | Обязательный |
| `--output, -o` | Путь для сохранения результата | Обязательный |
| `--api-key, -k` | API ключ | Из переменной OPENAI_API_KEY |
| `--api-base, -b` | URL API-сервера | https://cloud.m1r0.ru/v1 |
| `--model, -m` | Название модели | qwen3-coder:30b |
| `--max-concurrent, -c` | Максимум параллельных вызовов | 2 |
| `--retries, -r` | Количество попыток при ошибке | 3 |

### Использование Jupyter Notebook

```bash
jupyter notebook optim_Solution2.ipynb
```

## Дообучение модели

### Генерация датасета для обучения

```bash
# Создание базового датасета из входных данных
python finetune.py generate \
    --input ./Датасет/flights.json \
    --output ./training_data.jsonl
```

### Создание шаблона для ручной аннотации

```bash
# Создание шаблона для аннотации
python finetune.py annotate \
    --input ./Датасет/flights.json \
    --output ./annotations_template.json

# После заполнения аннотаций, конвертация в датасет
python finetune.py convert \
    --annotations ./annotations_filled.json \
    --output ./training_data_annotated.jsonl
```

### Запуск дообучения

```bash
# Установка дополнительных зависимостей для обучения
pip install torch transformers datasets peft trl bitsandbytes accelerate

# Дообучение с использованием QLoRA (4-bit)
python finetune.py train \
    --dataset ./training_data.jsonl \
    --output ./finetuned_model \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct

# Дообучение с кастомными параметрами
python finetune.py train \
    --dataset ./training_data.jsonl \
    --output ./finetuned_model \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 4 \
    --epochs 3 \
    --learning-rate 2e-4 \
    --max-seq-length 2048
```

### Объединение LoRA весов с базовой моделью

```bash
python finetune.py merge \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
    --lora-path ./finetuned_model \
    --output ./merged_model
```

### Параметры дообучения

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--dataset, -d` | Путь к датасету JSONL | Обязательный |
| `--output, -o` | Директория для модели | Обязательный |
| `--base-model, -m` | Базовая модель | Qwen/Qwen2.5-Coder-7B-Instruct |
| `--lora-r` | Ранг LoRA | 16 |
| `--lora-alpha` | Альфа параметр LoRA | 32 |
| `--batch-size` | Размер батча | 4 |
| `--epochs` | Количество эпох | 3 |
| `--learning-rate` | Скорость обучения | 2e-4 |
| `--max-seq-length` | Максимальная длина | 2048 |
| `--no-4bit` | Отключить 4-bit квантизацию | False |
| `--use-8bit` | Использовать 8-bit квантизацию | False |

## Структура проекта

```
LCT_2025/
├── run_solution.py      # Скрипт для запуска оптимизации
├── finetune.py          # Скрипт для дообучения модели
├── optim_Solution2.ipynb # Jupyter notebook с решением
├── requirements.txt     # Зависимости
├── README.md           # Документация
└── Датасет/            # Датасеты
    ├── flights.json    # Данные о рейсах
    └── questsH.json    # Данные о квестах
```

## Формат входных данных

```json
{
    "url": "jdbc:trino://...",
    "ddl": [
        {"statement": "CREATE TABLE ..."}
    ],
    "queries": [
        {
            "queryid": "uuid",
            "query": "SELECT ...",
            "runquantity": 100,
            "executiontime": 10
        }
    ]
}
```

## Формат выходных данных

```json
{
    "ddl": [
        {"statement": "CREATE TABLE ..."}
    ],
    "migrations": [
        {"statement": "INSERT INTO ..."}
    ],
    "queries": [
        {
            "queryid": "uuid",
            "query": "SELECT ..."
        }
    ]
}
```

## Лицензия

MIT
