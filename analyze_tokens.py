import argparse
from pathlib import Path
from token_utils import count_tokens, estimate_cost


def analyze_file(file_path: Path) -> None:
    """Анализирует текстовый файл и выводит метрики."""
    try:
        text = file_path.read_text(encoding="utf-8")  # UTF-8 для русского текста
    except FileNotFoundError:
        print(f"Файл '{file_path}' не найден.")
        return
    except UnicodeDecodeError:
        print(f"Не удалось прочитать '{file_path}' (проблема с кодировкой).")
        return

    num_chars = len(text)
    num_tokens = count_tokens(text, "gpt-4o-mini")

    chars_per_token = num_chars / num_tokens if num_tokens > 0 else 0

    print(f"\nАнализ файла: {file_path.name}")
    print("=" * 45)
    print(f"  Символов: {num_chars:,}")
    print(f"  Токенов (gpt-4o-mini): {num_tokens:,}")
    print(f"  Символов/токен: {chars_per_token:.1f}")

    models = ["gpt-4o-mini", "gpt-4o", "o3-mini"]
    print("\nОценка стоимости (completion = input):")
    print("-" * 45)
    print("Модель       | Input $  | Output $ | Total $")
    print("-" * 45)

    for model in models:
        cost = estimate_cost(num_tokens, num_tokens, model)
        print(
            f"{model:12} | {cost['input_cost']:7.6f} | {cost['output_cost']:7.6f} | "
            f"{cost['total_cost']:7.6f}"
        )
    print("-" * 45)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Анализирует токены и стоимость текстового файла."
    )
    parser.add_argument("file_path", type=Path, help="Путь к файлу")
    args = parser.parse_args()
    analyze_file(args.file_path)


if __name__ == "__main__":
    main()
