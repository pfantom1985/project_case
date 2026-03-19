import argparse
import sys
from pathlib import Path
from token_utils import count_tokens, estimate_cost


def analyze_file(file_path: Path) -> None:
    """Анализирует текстовый файл и выводит метрики."""
    try:
        text = file_path.read_text(encoding="utf-8") # Читаем файл с поддержкой UTF-8 (для русского текста)
    except FileNotFoundError:
        print(f"Файл '{file_path}' не найден.")
        return
    except UnicodeDecodeError:
        print(f"Не удалось прочитать файл '{file_path}' (проблема с кодировкой).")
        return

    num_chars = len(text)
    num_tokens_g4o_mini = count_tokens(text, "gpt-4o-mini")

    # Отношение символов к токенам (важно для языкового эксперимента)
    chars_per_token = num_chars / num_tokens_g4o_mini if num_tokens_g4o_mini > 0 else 0

    print(f"\nАнализ файла: {file_path.name}")
    print("=" * 60)
    print(f"  Символов: {num_chars:,}")
    print(f"  Токенов (gpt-4o-mini): {num_tokens_g4o_mini:,}")
    print(f"  Символов/токен: {chars_per_token:.1f}")

    # Оценка стоимости для 3 моделей (принимаем completion_tokens = input_tokens)
    models = ["gpt-4o-mini", "gpt-4o", "o3-mini"]
    print("\nОценка стоимости (completion_tokens = input_tokens):")
    print("Модель        | Input $  | Output $ | Total $")
    print("-" * 40)

    for model in models:
        cost = estimate_cost(num_tokens_g4o_mini, num_tokens_g4o_mini, model)
        print(
            f"{model:12} | {cost['input_cost']:7.6f} | {cost['output_cost']:7.6f} | "
            f"{cost['total_cost']:7.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Анализирует токены и стоимость обработки текстового файла."
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Путь к текстовому файлу для анализа",
    )

    args = parser.parse_args()
    analyze_file(args.file_path)


if __name__ == "__main__":
    main()
