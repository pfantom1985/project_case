import json
from typing import Dict, List
import tiktoken

# Актуальные цены OpenAI на март 2026 (USD за 1M токенов). Источник: https://openai.com/api/pricing/
MODEL_PRICES = {
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4o": {
        "input": 5.00,
        "output": 15.00,
    },
    "o3-mini": {
        "input": 3.00,
        "output": 12.00,
    },
}

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        print(f"Токенизатор для '{model}' не найден. Используем 'cl100k_base'.")

    tokens = encoding.encode(text)
    return len(tokens)

def estimate_cost(
        prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini"
) -> Dict[str, float]:
    """
    Оценивает стоимость запроса в USD для указанной модели.
    """
    if model not in MODEL_PRICES:
        raise ValueError(f"Цены для модели '{model}' не известны. Доступные: {list(MODEL_PRICES.keys())}")

    prices = MODEL_PRICES[model]
    input_cost = (prompt_tokens / 1_000_000) * prices["input"]
    output_cost = (completion_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost

    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
    }

def check_context_fit(
        messages: List[Dict], model: str = "gpt-4o-mini", max_output_tokens: int = 4096
) -> Dict:
    """
    Проверяет, помещается ли история сообщений в контекстное окно модели.

    Args:
        messages: Список сообщений [{"role": "user", "content": "..."}, ...]
        model: Название модели
        max_output_tokens: Максимум токенов для ответа (резерв)
    """
    CONTEXT_WINDOWS = {
        "gpt-4o-mini": 128_000,
        "gpt-4o": 128_000,
        "o3-mini": 200_000,
    }

    context_window = CONTEXT_WINDOWS.get(model, 128_000)

    # Сериализуем сообщения в единый текст для подсчёта токенов
    serialized_messages = json.dumps(messages, ensure_ascii=False)
    input_tokens = count_tokens(serialized_messages, model)
    available_for_output = context_window - input_tokens - max_output_tokens
    fits = available_for_output >= 0

    return {
        "input_tokens": input_tokens,
        "available_for_output": max(0, available_for_output),
        "fits": fits,
        "context_window": context_window,
    }


if __name__ == "__main__":
    # Тест count_tokens
    text = "Привет! Это тест токенизации."
    print(f"Токенов в '{text}': {count_tokens(text)}")

    # Тест estimate_cost
    cost = estimate_cost(1000, 500, "gpt-4o-mini")
    print(f"Стоимость: {cost}")

    # Тест check_context_fit
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    context = check_context_fit(messages)
    print(f"Контекст: {context}")
