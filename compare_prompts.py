import os
import csv
import json
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import Langfuse
from token_utils import count_tokens, estimate_cost

# Настройка клиентов
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Промпты
PROMPTS = {
    "v1": "Вы полезный помощник. Ответьте кратко и по существу.",
    "v2": """Вы — профессиональный эксперт по программированию на Python с 20-летним стажем.
Ваша цель — давать глубокие, структурированные и технически точные ответы.
Требования к ответу:
1. Используйте четкую структуру (заголовки, списки).
2. Обязательно приводите примеры кода с комментариями.
3. Соблюдайте академически-деловой, но понятный тон.
4. Упоминайте возможные "подводные камни" или лучшие практики (best practices).
5. Ответ должен быть полным, но не превращаться в книгу.
Ответ должен быть на русском языке."""
}


def get_answer(question: str, version: str):
    system_prompt = PROMPTS[version]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    # Трассировка
    trace = langfuse.trace(
        name="prompt-comparison-trace",
        tags=[f"prompt-{version}"],
        metadata={"prompt_version": version, "experiment": "prompt-comparison"}
    )

    generation = trace.generation(
        name="llm_call",
        model="gpt-4o-mini",
        input=messages,
        model_parameters={"temperature": 0.7}
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )

    answer = response.choices[0].message.content

    # Токены и цена
    input_tokens = count_tokens(json.dumps(messages), "gpt-4o-mini")
    output_tokens = count_tokens(answer, "gpt-4o-mini")
    cost = estimate_cost(input_tokens, output_tokens, "gpt-4o-mini")

    generation.end(
        output=answer,
        usage={"input": input_tokens, "output": output_tokens}
    )
    langfuse.flush()

    return input_tokens, output_tokens, cost["total_cost"]


# Основной цикл
with open("test_questions.txt", "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

results = []
print(f"{'Вопрос':<30} | {'Токены v1':<10} | {'Токены v2':<10} | {'Стоимость v1':<12} | {'Стоимость v2':<12}")
print("-" * 90)

for q in questions:
    in1, out1, cost1 = get_answer(q, "v1")
    in2, out2, cost2 = get_answer(q, "v2")

    print(f"{q[:30]:<30} | {in1 + out1:<10} | {in2 + out2:<10} | ${cost1:<11.6f} | ${cost2:<11.6f}")

    results.extend([
        [q, "v1", in1, out1, cost1],
        [q, "v2", in2, out2, cost2]
    ])

# Сохранение в CSV
with open("prompt_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Вопрос", "Версия", "Входные токены", "Выходные токены", "Общая стоимость"])
    writer.writerows(results)

print("\nРезультаты сохранены в prompt_comparison.csv")
