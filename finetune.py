#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для дообучения (fine-tuning) модели оптимизации SQL запросов.

Решение хакатона ЛЦТ 2025 - дообучение модели для оптимизации SQL запросов под Trino + Iceberg + S3.

Использование:
    # Генерация датасета для дообучения
    python finetune.py generate --input ./Датасет/flights.json --output ./training_data.jsonl

    # Дообучение модели (локально с использованием LoRA)
    python finetune.py train --dataset ./training_data.jsonl --output ./finetuned_model

    # Дообучение с указанием базовой модели
    python finetune.py train --dataset ./training_data.jsonl --output ./finetuned_model --base-model Qwen/Qwen2.5-Coder-7B-Instruct
"""

import argparse
import json
import os
import random
from typing import Any

# Промпты из основного решения
SYSTEM_PROMPTS = {
    "query_analyzer": """Ты — SQL-аналитик для системы оптимизации Data Lakehouse (Trino + Iceberg + S3).
Анализируй SQL-запросы и описывай структуру, паттерны, частые операции.
Указывай таблицы, фильтры, JOIN, GROUP BY, ORDER BY и возможные оптимизации через DDL.
Не используй materialized views, не придумывай статистику.""",

    "query_summarizer": """Ты — агрегирующий аналитик SQL для пайплайна оптимизации Data Lakehouse (Trino + Iceberg + S3).
Объединяй анализы в сводный отчёт: hot таблицы, частые JOIN, фильтры, тяжёлые операции.
Формируй рекомендации уровня DDL без генерации самого DDL.""",

    "ddl_optimizer": """Ты — DDL-оптимизатор для Trino + Iceberg + S3.
Создавай оптимизированные DDL на основе анализа запросов.
Используй партиционирование, денормализацию. Только валидный синтаксис Trino + Iceberg.""",

    "migrations_creator": """Ты — генератор миграций для Trino + Iceberg + S3.
Создавай безопасные SQL-миграции для применения DDL.
Порядок: создание таблиц, перенос данных, изменение свойств, валидация.""",

    "query_optimizer": """Ты — SQL-оптимизатор для Trino + Iceberg + S3.
Переписывай SQL запросы для использования новой структуры таблиц.
Сохраняй бизнес-логику, используй денормализованные таблицы, партиционные фильтры.""",

    "critic": """Ты — LLM-критик SQL для Trino + Iceberg.
Проверяй и исправляй ошибки в оптимизированных SQL: алиасы, существование колонок, 
JOIN-ключи, агрегации, синтаксис. Возвращай "OK" или исправленный SQL.""",

    "judge": """Ты — LLM-судья эквивалентности SQL (Trino).
Оценивай, сохраняет ли оптимизированный запрос бизнес-логику оригинала.
Возвращай "OK" или исправленный SQL с корректной семантикой.""",
}


def generate_training_data(input_path: str, output_path: str) -> None:
    """
    Генерация датасета для дообучения из исходных данных.

    Args:
        input_path: Путь к входному JSON файлу
        output_path: Путь для сохранения JSONL файла
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    training_examples: list[dict[str, Any]] = []

    ddl_statements = "\n".join([item["statement"] for item in data["ddl"]])

    for query_item in data["queries"]:
        query = query_item["query"]
        run_quantity = query_item.get("runquantity", 0)
        execution_time = query_item.get("executiontime", 0)

        # Пример для анализатора запросов
        training_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["query_analyzer"]},
                {
                    "role": "user",
                    "content": f"Проанализируй SQL запрос:\n{query}\n\n"
                    f"Количество выполнений: {run_quantity}\n"
                    f"Время выполнения: {execution_time} мс",
                },
                {
                    "role": "assistant",
                    "content": f"[PLACEHOLDER: Анализ запроса с указанием таблиц, фильтров, JOIN, операций и рекомендаций по DDL]",
                },
            ],
            "task_type": "query_analysis",
        })

        # Пример для оптимизатора запросов
        training_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["query_optimizer"]},
                {
                    "role": "user",
                    "content": f"DDL таблиц:\n{ddl_statements}\n\nОптимизируй запрос:\n{query}",
                },
                {
                    "role": "assistant",
                    "content": f"[PLACEHOLDER: Оптимизированный SQL запрос с использованием партиций и денормализации]",
                },
            ],
            "task_type": "query_optimization",
        })

        # Пример для критика
        training_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["critic"]},
                {
                    "role": "user",
                    "content": f"old_ddl:\n{ddl_statements}\n\n"
                    f"original:\n{query}\n\n"
                    f"optimized:\n{query}",
                },
                {"role": "assistant", "content": "OK"},
            ],
            "task_type": "query_critique",
        })

    # Примеры для DDL оптимизатора
    training_examples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["ddl_optimizer"]},
            {
                "role": "user",
                "content": f"Анализ запросов показал частые фильтры по дате и JOIN по ID.\n\n"
                f"Исходные DDL:\n{ddl_statements}",
            },
            {
                "role": "assistant",
                "content": "[PLACEHOLDER: Оптимизированные DDL с партиционированием]",
            },
        ],
        "task_type": "ddl_optimization",
    })

    # Сохранение в JSONL формате
    with open(output_path, "w", encoding="utf-8") as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Сгенерировано {len(training_examples)} примеров для обучения")
    print(f"Данные сохранены в {output_path}")
    print("\nВАЖНО: Замените [PLACEHOLDER: ...] на реальные примеры!")


def prepare_dataset_for_training(dataset_path: str) -> tuple[list, list]:
    """
    Подготовка датасета для обучения.

    Args:
        dataset_path: Путь к JSONL файлу

    Returns:
        Кортеж (train_data, eval_data)
    """
    data = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Перемешивание данных
    random.shuffle(data)

    # Разделение на train/eval (90/10)
    split_idx = int(len(data) * 0.9)
    return data[:split_idx], data[split_idx:]


def train_model(
    dataset_path: str,
    output_dir: str,
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    use_4bit: bool = True,
    use_8bit: bool = False,
) -> None:
    """
    Дообучение модели с использованием LoRA/QLoRA.

    Args:
        dataset_path: Путь к датасету в формате JSONL
        output_dir: Директория для сохранения модели
        base_model: Название базовой модели
        lora_r: Ранг LoRA
        lora_alpha: Альфа параметр LoRA
        lora_dropout: Dropout для LoRA
        batch_size: Размер батча
        gradient_accumulation_steps: Шаги накопления градиента
        num_epochs: Количество эпох
        learning_rate: Скорость обучения
        max_seq_length: Максимальная длина последовательности
        use_4bit: Использовать 4-bit квантизацию (QLoRA)
        use_8bit: Использовать 8-bit квантизацию
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("\nУстановите необходимые библиотеки:")
        print("pip install torch transformers datasets peft trl bitsandbytes accelerate")
        return

    print(f"Загрузка датасета из {dataset_path}...")
    train_data, eval_data = prepare_dataset_for_training(dataset_path)

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        """Форматирование примера в текст."""
        messages = example["messages"]
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|system|>\n{content}\n"
            elif role == "user":
                text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}\n"
        return {"text": text}

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Настройка квантизации
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Загрузка модели {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Подготовка модели для обучения
    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)

    # Настройка LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Настройка обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Создание тренера
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Начало обучения...")
    trainer.train()

    print(f"Сохранение модели в {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Обучение завершено!")
    print(f"Модель сохранена в {output_dir}")
    print("\nДля использования модели:")
    print(f"  from peft import PeftModel, PeftConfig")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  ")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{base_model}')")
    print(f"  model = PeftModel.from_pretrained(model, '{output_dir}')")


def merge_lora_weights(
    base_model: str,
    lora_model_path: str,
    output_dir: str,
) -> None:
    """
    Объединение LoRA весов с базовой моделью.

    Args:
        base_model: Название базовой модели
        lora_model_path: Путь к LoRA модели
        output_dir: Директория для сохранения объединённой модели
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("\nУстановите необходимые библиотеки:")
        print("pip install transformers peft")
        return

    print(f"Загрузка базовой модели {base_model}...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Загрузка LoRA весов из {lora_model_path}...")
    model = PeftModel.from_pretrained(base, lora_model_path)

    print("Объединение весов...")
    model = model.merge_and_unload()

    print(f"Сохранение модели в {output_dir}...")
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print("Объединение завершено!")
    print(f"Модель сохранена в {output_dir}")


def create_example_annotations(input_path: str, output_path: str) -> None:
    """
    Создание шаблона для ручной аннотации примеров.

    Args:
        input_path: Путь к входному JSON файлу
        output_path: Путь для сохранения шаблона
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    annotations: list[dict[str, Any]] = []

    ddl_statements = "\n".join([item["statement"] for item in data["ddl"]])

    for query_item in data["queries"]:
        query = query_item["query"]
        query_id = query_item["queryid"]
        run_quantity = query_item.get("runquantity", 0)
        execution_time = query_item.get("executiontime", 0)

        annotations.append({
            "query_id": query_id,
            "original_query": query,
            "run_quantity": run_quantity,
            "execution_time": execution_time,
            "ddl_context": ddl_statements,
            "annotations": {
                "analysis": "TODO: Добавьте анализ запроса",
                "optimized_query": "TODO: Добавьте оптимизированный запрос",
                "optimization_explanation": "TODO: Объясните оптимизацию",
            },
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"Шаблон для аннотации сохранён в {output_path}")
    print(f"Всего {len(annotations)} запросов для аннотации")


def convert_annotations_to_training_data(
    annotations_path: str, output_path: str
) -> None:
    """
    Конвертация аннотированных примеров в формат для обучения.

    Args:
        annotations_path: Путь к JSON файлу с аннотациями
        output_path: Путь для сохранения JSONL файла
    """
    with open(annotations_path, encoding="utf-8") as f:
        annotations = json.load(f)

    training_examples: list[dict[str, Any]] = []

    for item in annotations:
        if "TODO:" in str(item["annotations"]):
            print(f"Пропуск неаннотированного примера: {item['query_id']}")
            continue

        query = item["original_query"]
        ddl = item["ddl_context"]
        analysis = item["annotations"]["analysis"]
        optimized = item["annotations"]["optimized_query"]
        run_quantity = item.get("run_quantity", 0)
        execution_time = item.get("execution_time", 0)

        # Пример для анализатора
        training_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["query_analyzer"]},
                {
                    "role": "user",
                    "content": f"Проанализируй SQL запрос:\n{query}\n\n"
                    f"Количество выполнений: {run_quantity}\n"
                    f"Время выполнения: {execution_time} мс",
                },
                {"role": "assistant", "content": analysis},
            ],
            "task_type": "query_analysis",
        })

        # Пример для оптимизатора
        training_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["query_optimizer"]},
                {
                    "role": "user",
                    "content": f"DDL таблиц:\n{ddl}\n\nОптимизируй запрос:\n{query}",
                },
                {"role": "assistant", "content": optimized},
            ],
            "task_type": "query_optimization",
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Сконвертировано {len(training_examples)} примеров")
    print(f"Данные сохранены в {output_path}")


def main() -> None:
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Дообучение модели оптимизации SQL запросов (ЛЦТ 2025)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # Команда generate
    gen_parser = subparsers.add_parser(
        "generate", help="Генерация датасета для дообучения"
    )
    gen_parser.add_argument(
        "--input", "-i", required=True, help="Путь к входному JSON файлу"
    )
    gen_parser.add_argument(
        "--output", "-o", required=True, help="Путь для сохранения JSONL файла"
    )

    # Команда train
    train_parser = subparsers.add_parser("train", help="Дообучение модели")
    train_parser.add_argument(
        "--dataset", "-d", required=True, help="Путь к датасету в формате JSONL"
    )
    train_parser.add_argument(
        "--output", "-o", required=True, help="Директория для сохранения модели"
    )
    train_parser.add_argument(
        "--base-model",
        "-m",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Базовая модель для дообучения",
    )
    train_parser.add_argument(
        "--lora-r", type=int, default=16, help="Ранг LoRA"
    )
    train_parser.add_argument(
        "--lora-alpha", type=int, default=32, help="Альфа параметр LoRA"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=4, help="Размер батча"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=3, help="Количество эпох"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Скорость обучения"
    )
    train_parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Максимальная длина"
    )
    train_parser.add_argument(
        "--no-4bit", action="store_true", help="Отключить 4-bit квантизацию"
    )
    train_parser.add_argument(
        "--use-8bit", action="store_true", help="Использовать 8-bit квантизацию"
    )

    # Команда merge
    merge_parser = subparsers.add_parser(
        "merge", help="Объединение LoRA весов с базовой моделью"
    )
    merge_parser.add_argument(
        "--base-model",
        "-m",
        required=True,
        help="Базовая модель",
    )
    merge_parser.add_argument(
        "--lora-path", "-l", required=True, help="Путь к LoRA модели"
    )
    merge_parser.add_argument(
        "--output", "-o", required=True, help="Директория для сохранения"
    )

    # Команда annotate
    annotate_parser = subparsers.add_parser(
        "annotate", help="Создание шаблона для аннотации"
    )
    annotate_parser.add_argument(
        "--input", "-i", required=True, help="Путь к входному JSON файлу"
    )
    annotate_parser.add_argument(
        "--output", "-o", required=True, help="Путь для сохранения шаблона"
    )

    # Команда convert
    convert_parser = subparsers.add_parser(
        "convert", help="Конвертация аннотаций в датасет"
    )
    convert_parser.add_argument(
        "--annotations", "-a", required=True, help="Путь к файлу с аннотациями"
    )
    convert_parser.add_argument(
        "--output", "-o", required=True, help="Путь для сохранения JSONL"
    )

    args = parser.parse_args()

    if args.command == "generate":
        generate_training_data(args.input, args.output)
    elif args.command == "train":
        train_model(
            dataset_path=args.dataset,
            output_dir=args.output,
            base_model=args.base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            use_4bit=not args.no_4bit,
            use_8bit=args.use_8bit,
        )
    elif args.command == "merge":
        merge_lora_weights(
            base_model=args.base_model,
            lora_model_path=args.lora_path,
            output_dir=args.output,
        )
    elif args.command == "annotate":
        create_example_annotations(args.input, args.output)
    elif args.command == "convert":
        convert_annotations_to_training_data(args.annotations, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
