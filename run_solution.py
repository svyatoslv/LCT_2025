#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для запуска мультиагентного пайплайна оптимизации SQL запросов.

Решение хакатона ЛЦТ 2025 - оптимизация SQL запросов для Trino + Iceberg + S3.

Использование:
    python run_solution.py --input ./Датасет/flights.json --output ./output_result.json
    python run_solution.py --input ./Датасет/flights.json --output ./output_result.json --api-key YOUR_API_KEY
"""

import argparse
import asyncio
import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


def normalize_sql(sql: str) -> str:
    """
    Нормализует SQL запрос: заменяет множественные пробелы и переносы на одиночные пробелы.
    Сохраняет структуру запроса, удаляя лишние пробелы.
    """
    # Replace newlines and multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', sql)
    return normalized.strip()


def query_analize_prompt() -> str:
    """Промпт для анализа SQL-запросов."""
    return """Ты — SQL-аналитик для системы оптимизации Data Lakehouse (Trino + Iceberg + S3).
Вход: до 5 SQL-запросов.  
Твоя цель — описать словами структуру и частые паттерны этих запросов, указав:
- какие таблицы чаще всего участвуют;
- какие поля часто используются в фильтрах (WHERE/ON);
- по каким полям часто происходит соединение (JOIN);
- какие операции чаще всего выполняются (GROUP BY, ORDER BY, DISTINCT, WINDOW);
- какие конструкции создают нагрузку или могут быть оптимизированы через DDL;
- какие улучшения можно предложить на уровне DDL (партиционирование, денормализация, сортировка и т. п.).

### Важно:
0. Обращай внимание на частоту выполнения запроса и его время, чтобы в первую очередь оптимизировать тяжёлые запросы
1. **Запрещено** предлагать или использовать materialized views.
2. **Запрещено** менять DDL или писать SQL-код миграций. Только описывать, что требует оптимизации.  
3. **Запрещено** придумывать статистику, размеры таблиц или время выполнения. Если данных нет — укажи "неизвестно".  
4. **Не упоминай** индексы в классическом RDBMS-смысле (B-Tree и т. д.).  
5. **Не упоминай** безопасность, авторизацию, шифрование и внешние системы.  
6. Используй формулировки, применимые к Trino + Iceberg + S3.  
   (Например: «можно рассмотреть партиционирование по дате» вместо «создать индекс».)  
7. Не возвращай JSON, таблицы или списки ключей. Пиши связный текстовый отчёт с пунктами и примерами.

### Структура ответа:
- Краткое резюме: какие таблицы и поля чаще всего встречаются.  
- Анализ соединений: какие таблицы чаще соединяются между собой, по каким ключам.  
- Анализ фильтров: какие колонки часто участвуют в WHERE/ON (например, event_date, user_id).  
- Анализ операций: какие операции создают нагрузку (JOIN, GROUP BY, DISTINCT, ORDER BY, WINDOW).

Вывод должен быть понятным, логически структурированным и готовым для последующей обработки другой моделью, которая будет строить DDL и миграции на основе твоего анализа."""


def query_summarize_prompt() -> str:
    """Промпт для суммирования результатов анализа."""
    return """Ты — агрегирующий аналитик SQL для пайплайна оптимизации Data Lakehouse (стек: Trino + Iceberg + S3).

ВХОД:
- Строка / текст `analize` — объединённые результаты работы query_analize_agent для многих батчей (каждый блок содержит анализ до 5 SQL-запросов).
- Формат входа может быть JSON-подобным или текстовым отчётом (см. предыдущий агент). Разбирай оба варианта; если не можешь распарсить фрагмент — помечь как "UNPARSED fragment".

ЦЕЛЬ:
- Объединить и агрегировать мелкие анализы в единый сводный отчёт, выявив:
  1. "Hot" таблицы — таблицы с наибольшей частотой упоминаний в тяжёлых операциях;
  2. Частые пары JOIN (и их join_keys);
  3. Часто используемые фильтры (колонки в WHERE/ON);
  4. Частые тяжёлые операции (GROUP BY, ORDER BY, DISTINCT, WINDOW, SORT);
  5. Повторяющиеся паттерны, которые можно исправить DDL-изменениями;
  6. Очередность (приоритет) изменений для downstream: DDL_agent -> migrations_agent -> query_optimizer.
- Сформировать короткий и однозначный набор рекомендаций уровня DDL (НЕ писать сам DDL/миграции, а только что и где изменить и почему).

СТРОГИЕ ЗАПРЕЩЕНИЯ (выполняй обязательно):
- НЕЛЬЗЯ предлагать или использовать materialized views.
- НЕЛЬЗЯ писать, изменять или генерировать DDL или миграции — это делают downstream агенты.
- НЕЛЬЗЯ придумывать статистику, объёмы данных, кардинальности или время выполнения; если этих метрик нет — указывай `UNKNOWN`.
- НЕЛЬЗЯ предлагать классические RDBMS-индексы (B-Tree и т.п.). Если предлагаешь что-то похожее, опиши его как "файловая/партиц./сортировка/кластеризация в Iceberg" и пометь `depends_on_iceberg_features`.
- НЕЛЬЗЯ обсуждать авторизацию/аутентификацию/безопасность/внешние сервисы.
- Ответ должен быть ТОЛЬКО текстом отчёта (без генерации JSON/DDL/SQL). Пиши структурированный человекочитаемый отчёт (см. формат ниже).

ПРАВИЛА АГРЕГАЦИИ:
- Подсчитывай частоту встречаемости сущностей (таблиц, колонок, join-пар) по данным `analize`. Если входы не содержат явного счётчика — используй относительную частоту (high/medium/low) основанную на количестве вхождений в анализах; если нельзя определить — ставь `UNKNOWN`.
- Если множество анализов указывает на фильтрацию по колонке `event_date`/`ds` — предлагай PARTITIONING (указывай день/месяц как опции). Добавляй замечание о риске при высокой кардинальности.
- Если одна и та же пара таблиц соединяется очень часто — размышляй о DENORMALIZATION (опиши, какие поля включить), но НЕ генерируй CREATE TABLE.
- Для ORDER BY + LIMIT — указывай возможность SORT_ORDER / предварительной кластеризации в Iceberg и помечай `depends_on_iceberg_features`.
- В каждой рекомендации указывай: цель (что менять), конкретные колонки/таблицы, краткая причина (основанную на evidence fragments из `analize`), ориентировочный impact (HIGH/MEDIUM/LOW/UNKNOWN) и какие метрики/проверки нужны (data_volume, cardinality, query_frequency, avg_runtime).
"""


def ddl_optim_prompt() -> str:
    """Промпт для оптимизации DDL."""
    return """Ты — DDL-оптимизатор для Trino + Iceberg + S3.

Вход:
- query_summarize: анализ частых фильтров, сортировок, JOIN
- ddl_orig: исходные DDL таблиц

Задача: Создать оптимизированные DDL для Trino + Iceberg.

СИНТАКСИС Trino + Iceberg:
- Таблицы: <catalog>.<schema>.<table>
- Формат хранения файлов: используйте WITH (format = 'PARQUET')
- Партиционирование (Trino/Iceberg): в CREATE TABLE через WITH (partitioning = ARRAY['day(ts_col)'] или 'month(ts_col)'). Не комбинировать year/month/day на одном столбце.
- Свойства: только допустимые Trino/Iceberg ключи в WITH (...). Не использовать 'write.target-file-size-bytes'. Без висячих запятых.

РАЗРЕШЕНО:
- PARTITIONING через WITH с допустимыми функциями
- CREATE TABLE AS SELECT WITH (format = 'PARQUET')
- Создание оптимизированных копий

ЗАПРЕЩЕНО:
- Materialized Views, индексы
- DROP, DELETE, RENAME
- Несовместимый синтаксис

ПРИМЕРЫ:
CREATE TABLE catalog.schema.table_new WITH (partitioning = ARRAY['year(order_date)']) AS SELECT * FROM catalog.schema.table_old;
CREATE TABLE analytics.sales.orders_new AS SELECT * FROM orders WITH (format = 'PARQUET');

ПРАВИЛА ВЫВОДА:
- Каждый оператор — отдельной строкой; не объединяй множество операторов в одну строку.
- Полные имена таблиц (<catalog>.<schema>.<table>) согласованы с JDBC url.
- Никаких комментариев, префиксов типа "sql ". Только чистые SQL-операторы.

ВЫВОД: Только SQL DDL команды, готовые к выполнению в Trino."""


def migrations_creator_prompt() -> str:
    """Промпт для создания миграций."""
    return """Ты — генератор миграций для Trino + Iceberg + S3.

ВХОД:
- query_summarize: анализ использования таблиц
- new_ddl: оптимизированные DDL-запросы

ЦЕЛЬ:
Сгенерировать безопасные SQL-миграции для применения DDL в Trino + Iceberg.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА СИНТАКСИСА:
1. ВСЕ таблицы: <catalog>.<schema>.<table>
2. ТОЛЬКО совместимый с Trino + Iceberg синтаксис
3. НЕТ materialized views, индексов, деструктивных операций
4. Формат файлов: используем 'PARQUET' через WITH (format = 'PARQUET')

РАЗРЕШЕННЫЕ ОПЕРАЦИИ МИГРАЦИИ:
- Не дублировать DDL из new_ddl: миграции не должны повторять CREATE TABLE/CTAS, уже присутствующие в new_ddl.
- CREATE TABLE ... WITH (format = 'PARQUET', partitioning = ARRAY[...])
- ALTER TABLE ... PARTITIONING (через WITH) ... (только допустимые функции: year/month/day/hour/bucket/truncate)
- INSERT INTO new_table SELECT FROM old_table
- CREATE TABLE ... AS SELECT ... WITH (format = 'PARQUET')

ПОРЯДОК МИГРАЦИЙ:
1. Создание новых таблиц с оптимизациями
2. Перенос данных (если нужно)
3. Изменение свойств существующих таблиц
4. Валидация (простая проверка COUNT)

ПРИМЕРЫ ВАЛИДНОГО СИНТАКСИСА:
CREATE TABLE analytics.sales.orders_new
WITH (format = 'PARQUET', partitioning = ARRAY['day(order_date)'])
AS SELECT * FROM analytics.sales.orders;

ALTER TABLE analytics.sales.orders
PARTITIONING (через WITH) year(order_date);

ALTER TABLE analytics.sales.orders
-- removed unsafe SET PROPERTIES example

ВЫВОД:
ТОЛЬКО SQL-команды миграций, готовые к выполнению в Trino.
Без комментариев, пояснений, текста.
Каждая команда на новой строке."""


def query_optimize_prompt() -> str:
    """Промпт для оптимизации SQL-запросов."""
    return """Ты — SQL-оптимизатор для Trino + Iceberg + S3.

ВХОД:
- new_ddl: новые DDL таблиц (партиции, денормализации, свойства)
- query: исходный SQL запрос для оптимизации

ЦЕЛЬ:
Переписать SQL запрос для использования новой структуры таблиц из DDL.
Сохранить идентичную бизнес-логику и результат.

ПРАВИЛА ОПТИМИЗАЦИИ:
1. Используй новые таблицы и колонки из DDL
2. Для денормализованных таблиц - убирай JOIN, используй прямые обращения
3. Для партиционированных таблиц - фильтруй по полям партиций
4. Сохраняй все агрегации, фильтры и логику оригинала
5. Используй только синтаксис Trino + Iceberg

ЗАПРЕЩЕНО:
- Менять бизнес-логику (WHERE, JOIN, GROUP BY кроме адаптации к DDL)
- Materialized Views, индексы, временные таблицы
- Комментарии, пояснения, не-SQL текст

ПРИМЕР:
Исходный: SELECT u.id, SUM(o.amount) FROM orders o JOIN users u ON o.user_id = u.id
DDL: CREATE TABLE orders_denorm AS SELECT o.*, u.region FROM orders o JOIN users u ON o.user_id = u.id
Оптимизированный: SELECT user_id, SUM(amount) FROM orders_denorm

АНТИ-ОШИБКИ (соблюдать обязательно):
- Каждый источник в FROM/JOIN обязан иметь алиас; все колонки должны быть квалифицированы этим алиасом.
- Нельзя ссылаться на алиас другой таблицы (пример: pc.client_id, если в FROM pc= l_excursion_payment, где нет client_id). Выбирай корректные поля исходной таблицы или корректируй JOIN.
- Любая подзапрос/CTE обязан иметь алиас (.. ) AS sub; не оставлять безымянные скобки.
- GROUP BY: перечисляй только реально присутствующие в SELECT выражения/поля; не добавлять лишние токены (например, 'b1').
- Скобки и запятые: не оставлять висячих запятых и лишних закрывающих скобок.
- ORDER BY random() использовать только если это требуется задачей; избегать недетерминизма.

ВЫВОД: Только оптимизированный SQL запрос для Trino, без комментариев."""


def critic_prompt() -> str:
    """Промпт для LLM-критика SQL."""
    return """Ты — LLM‑критик SQL для Trino + Iceberg.

ВХОД (plain‑text блоки, без JSON):

old_ddl:
<CREATE TABLE ...>  (по одному на строку; опционально)

new_ddl:
<CREATE TABLE ...>  (по одному на строку; опционально)

migrations:
<CTAS/INSERT INTO ... SELECT ...>  (перенос из старой структуры в новую; опционально)

original:
<исходный SQL>

optimized:
<оптимизированный SQL>

ЦЕЛЬ:
- Самостоятельно найти и исправить ошибки в поле optimized, если они конечно присутствуют. Проведи собственную многошаговую проверку консистентности, используя при наличии DDL (ddl_old/ddl_new) как источник истины по колонкам и таблицам.
- Исходный запрос (original) практически всегда валиден. Разрешено переиспользовать его конструкции (алиасы, выражения, подзапросы, группировки) во втором запросе, но обязательно с учётом уже выполненной оптимизации (например, денормализация, замена источников, добавление партиционных фильтров).
- Проверь строго по чек‑листу:
  1) Алиасы: каждый источник в FROM/JOIN имеет алиас; все колонки квалифицированы правильным алиасом.
  2) Существование колонок: alias.column существует согласно DDL соответствующей таблицы; при денормализации используй новые таблицы из ddl_new. Если DDL отсутствуют — делай правки только по очевидным несоответствиям (алиасы/скобки/Group By), не придумывай колонки.
  3) JOIN‑ключи: ссылки только на колонки, реально присутствующие в соответствующих таблицах; не используй client_id у таблицы, где его нет.
  4) Агрегации: GROUP BY соответствует SELECT (все неагрегированные выражения перечислены), нет мусорных токенов (например, 'b1').
  5) Синтаксис: нет лишних/незакрытых скобок, висячих запятых.
  6) Детерминизм: избегай ORDER BY random() без явной необходимости.
- Если оптимизация меняла структуру: корректно перепиши обращения к колонкам и JOIN с учётом ddl_new.
- Если ошибок НЕТ — верни строку "OK" (без дополнительных слов).
- Если ошибки ЕСТЬ — верни ТОЛЬКО один исправленный SQL‑запрос, без комментариев и текста.

ОБЯЗАТЕЛЬНО:
- Каждый источник в FROM/JOIN должен иметь алиас.
- Все колонки должны быть квалифицированы правильным алиасом.
- Не ссылаться на поля, которых нет в источнике (пример: pc.client_id, если pc= l_excursion_payment, где нет client_id).
- Каждый подзапрос/CTE обязан иметь алиас.
- GROUP BY: перечисляй только выражения/поля из SELECT; не добавляй посторонние токены.
- Скобки и запятые: без висячих запятых и лишних скобок.
- Использовать синтаксис Trino; таблицы — в формате <catalog>.<schema>.<table>.
- Избегать ORDER BY random(), если это не требуется явно.

ВЫВОД:
- Если ошибок нет: ровно "OK".
- Если есть ошибки: только исправленный SQL.
"""


def judge_prompt() -> str:
    """Промпт для LLM-судьи эквивалентности SQL."""
    return """Ты — LLM‑судья эквивалентности SQL (Trino).

ВХОД (plain‑text блоки, без JSON):

old_ddl:
<CREATE TABLE ...>  (по одному на строку; опционально)

new_ddl:
<CREATE TABLE ...>  (по одному на строку; опционально)

migrations:
<CTAS/INSERT INTO ... SELECT ...>  (перенос из старой в новую; опционально)

original:
<исходный SQL>

optimized:
<оптимизированный SQL>

ЗАДАЧА:
- По структуре SQL (без выполнения) оцени, сохраняет ли optimized бизнес‑логику original (те же агрегаты, фильтры, соединения, проекции) с допустимыми адаптациями под оптимизацию (денормализация, партиционные фильтры и т. п.).
- Учитывай DDL/миграции: если optimized читает данные из новых таблиц (по CTAS/INSERT), интерпретируй соответствия полей между старой и новой структурой.

ВЫВОД:
- Если ошибок нет: ровно "OK".
- Если есть проблема (семантика/структура/синтаксис): верни полный исправленный SQL (один запрос, без комментариев/объяснений).
"""


class SQLOptimizationPipeline:
    """Мультиагентный пайплайн для оптимизации SQL запросов."""

    def __init__(
        self,
        api_base: str = "https://cloud.m1r0.ru/v1",
        api_key: str | None = None,
        model: str = "qwen3-coder:30b",
        max_concurrent_calls: int = 2,
        default_retry_nums: int = 3,
    ):
        """
        Инициализация пайплайна.

        Args:
            api_base: URL API-сервера
            api_key: API ключ
            model: Название модели
            max_concurrent_calls: Максимальное количество параллельных вызовов
            default_retry_nums: Количество попыток при ошибке
        """
        self.api_base = api_base
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.max_concurrent_calls = max_concurrent_calls
        self.default_retry_nums = default_retry_nums

        # Инициализация LLM
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            temperature=0,
        )

        # Инициализация агентов
        self._init_agents()

        # Семафор для контроля параллельных вызовов
        self.semaphore = asyncio.Semaphore(self.max_concurrent_calls)

        # Контекст для проверки запросов
        self.context_info: dict[str, Any] = {}

    def _init_agents(self) -> None:
        """Инициализация всех агентов."""
        self.query_analize_agent = create_react_agent(
            self.llm, tools=[], prompt=query_analize_prompt()
        )
        self.query_summarize_agent = create_react_agent(
            self.llm, tools=[], prompt=query_summarize_prompt()
        )
        self.ddl_optimize_agent = create_react_agent(
            self.llm, tools=[], prompt=ddl_optim_prompt()
        )
        self.migrations_creator_agent = create_react_agent(
            self.llm, tools=[], prompt=migrations_creator_prompt()
        )
        self.query_optimize_agent = create_react_agent(
            self.llm, tools=[], prompt=query_optimize_prompt()
        )
        self.judge_agent = create_react_agent(
            self.llm, tools=[], prompt=judge_prompt()
        )
        self.critic_agent = create_react_agent(
            self.llm, tools=[], prompt=critic_prompt()
        )

    async def _get_answer_async(
        self, agent: Any, content: str, retry_nums: int | None = None
    ) -> str:
        """Асинхронный вызов агента с повторными попытками."""
        retry_nums = retry_nums or self.default_retry_nums
        for _ in range(retry_nums):
            try:
                prompt = HumanMessage(content=content)
                response = await agent.ainvoke({"messages": prompt})
                return response["messages"][-1].content
            except Exception as e:
                print(f"Failed to execute. Error: {e!r}")
                print("Retrying...")
                await asyncio.sleep(1)
        print("ERROR: Model call failed after all retries")
        return "ERROR: Model call failed after all retries"

    def _get_answer(self, agent: Any, content: str) -> str:
        """Синхронный вызов агента с повторными попытками."""
        for _ in range(self.default_retry_nums):
            try:
                prompt = HumanMessage(content=content)
                return agent.invoke({"messages": prompt})["messages"][-1].content
            except Exception as e:
                print(f"Failed to execute. Error: {e!r}")
                print("Retrying...")
        print("ERROR: Model call failed after all retries")
        return "ERROR: Model call failed after all retries"

    async def _bounded_get_answer_async(self, agent: Any, content: str) -> str:
        """Асинхронный вызов с ограничением параллелизма."""
        async with self.semaphore:
            return await self._get_answer_async(agent, content)

    async def _check_query(self, old_query: str, new_query: str) -> str:
        """Проверка оптимизированного запроса критиком и судьёй."""
        prompt_content = (
            f"старые DDL запросы: \n{self.context_info['old_ddl']}\n\n"
            f"Новые DDL запросы: \n{self.context_info['new_ddl']}\n\n"
            f"Вот миграции:\n{self.context_info['migrations']}"
            f"оригинальный sql запрос:\n{old_query}"
            f"оптимизированный запрос:\n{new_query}"
        )

        # Проверка судьёй
        judge_response = await self._get_answer_async(
            self.judge_agent, prompt_content
        )
        if "OK" not in judge_response:
            new_query = judge_response

        # Проверка критиком
        prompt_content = (
            f"старые DDL запросы: \n{self.context_info['old_ddl']}\n\n"
            f"Новые DDL запросы: \n{self.context_info['new_ddl']}\n\n"
            f"Вот миграции:\n{self.context_info['migrations']}"
            f"оригинальный sql запрос:\n{old_query}"
            f"оптимизированный запрос:\n{new_query}"
        )

        critic_response = await self._get_answer_async(
            self.critic_agent, prompt_content
        )
        if "OK" not in critic_response:
            new_query = critic_response

        return new_query

    async def _optimize_single_query(self, new_ddl: str, query: str) -> str:
        """Оптимизация одного SQL запроса."""
        async with self.semaphore:
            for _ in range(self.default_retry_nums):
                try:
                    content = (
                        f"Новые DDL запросы: \n{new_ddl}\n\n"
                        f"Вот SQL запрос, который надо оптимизировать:\n{query}"
                    )
                    prompt = HumanMessage(content=content)
                    response = await self.query_optimize_agent.ainvoke(
                        {"messages": prompt}
                    )
                    optimized = response["messages"][-1].content
                    return await self._check_query(query, optimized)
                except Exception as e:
                    print(f"Failed to optimize query: {query[:50]}... Error: {e!r}")
                    print("Retrying...")
                    await asyncio.sleep(1)
            print(f"ERROR: Query optimization failed after all retries")
            return query  # Return original query if optimization fails

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Запуск полного пайплайна оптимизации.

        Args:
            input_data: Входные данные с DDL и запросами

        Returns:
            Результат с оптимизированными DDL, миграциями и запросами
        """
        # Извлечение данных
        ddl_queries = "\n".join([item["statement"] for item in input_data["ddl"]])
        sql_queries = [item["query"] for item in input_data["queries"]]

        # Шаг 1: Анализ запросов батчами
        print("Шаг 1: Анализ SQL запросов...")
        queries_batch_prompt = [
            f"sql-запрос: {item['query']}, количество выполнения:{item['runquantity']}, "
            f"количество затраченного времени при едином выполнении запроса: {item['executiontime']}"
            for item in input_data["queries"]
        ]

        batches = [
            queries_batch_prompt[5 * i : 5 * (i + 1)]
            for i in range(
                len(queries_batch_prompt) // 5
                + int(len(queries_batch_prompt) % 5 != 0)
            )
        ]

        tasks = [
            self._bounded_get_answer_async(
                self.query_analize_agent, "\n\n".join(batch)
            )
            for batch in batches
        ]
        results = await asyncio.gather(*tasks)
        analysis = "\n\n".join(results)
        print("Анализ завершён.")

        # Шаг 2: Суммирование анализа
        print("Шаг 2: Суммирование анализа...")
        query_summarize = self._get_answer(
            self.query_summarize_agent,
            f"Анализ полученных sql запросов: {analysis}",
        )
        print("Суммирование завершено.")

        # Шаг 3: Оптимизация DDL
        print("Шаг 3: Оптимизация DDL...")
        new_ddl = self._get_answer(
            self.ddl_optimize_agent,
            f"Анализ полученных sql запросов:\n{query_summarize} \n\n"
            f"это оригинальные DDL запросы: \n{ddl_queries}",
        )
        print("Оптимизация DDL завершена.")

        # Шаг 4: Создание миграций
        print("Шаг 4: Создание миграций...")
        migrations = self._get_answer(
            self.migrations_creator_agent,
            f"Анализ полученных sql запросов:\n{query_summarize} \n\n"
            f"это Новые DDL запросы: \n{new_ddl}",
        )
        print("Создание миграций завершено.")

        # Сохранение контекста для проверки запросов
        self.context_info = {
            "old_ddl": ddl_queries,
            "new_ddl": new_ddl,
            "migrations": migrations,
        }

        # Шаг 5: Оптимизация SQL запросов
        print("Шаг 5: Оптимизация SQL запросов...")
        optimization_tasks = [
            self._optimize_single_query(new_ddl, query) for query in sql_queries
        ]
        optimized_results = await asyncio.gather(*optimization_tasks)
        print("Оптимизация запросов завершена.")

        # Формирование результата
        optimized_queries = {
            input_data["queries"][i]["queryid"]: optimized_results[i]
            for i in range(len(optimized_results))
        }

        result = {
            "ddl": input_data["ddl"]
            + [
                {"statement": normalize_sql(stmt)}
                for stmt in new_ddl.split(";")
                if stmt.strip()
            ],
            "migrations": [
                {"statement": normalize_sql(stmt)}
                for stmt in migrations.split(";")
                if stmt.strip()
            ],
            "queries": [
                {
                    "queryid": item["queryid"],
                    "query": normalize_sql(optimized_queries[item["queryid"]]),
                }
                for item in input_data["queries"]
            ],
        }

        return result


def load_input_data(input_path: str) -> dict[str, Any]:
    """Загрузка входных данных из JSON файла."""
    with open(input_path, encoding="utf-8") as f:
        return json.load(f)


def save_output_data(output_path: str, data: dict[str, Any]) -> None:
    """Сохранение результата в JSON файл."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


async def main() -> None:
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Мультиагентный пайплайн оптимизации SQL запросов (ЛЦТ 2025)"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Путь к входному JSON файлу с DDL и запросами",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Путь для сохранения результата",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        help="API ключ (или установите OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--api-base",
        "-b",
        default="https://cloud.m1r0.ru/v1",
        help="URL API-сервера",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="qwen3-coder:30b",
        help="Название модели",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=2,
        help="Максимальное количество параллельных вызовов",
    )
    parser.add_argument(
        "--retries",
        "-r",
        type=int,
        default=3,
        help="Количество попыток при ошибке",
    )

    args = parser.parse_args()

    # Загрузка данных
    print(f"Загрузка данных из {args.input}...")
    input_data = load_input_data(args.input)

    # Создание и запуск пайплайна
    pipeline = SQLOptimizationPipeline(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_concurrent_calls=args.max_concurrent,
        default_retry_nums=args.retries,
    )

    print("Запуск пайплайна оптимизации...")
    result = await pipeline.run(input_data)

    # Сохранение результата
    save_output_data(args.output, result)
    print(f"Результат сохранён в {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
