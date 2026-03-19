import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, APIError


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Переменная окружения OPENAI_API_KEY не найдена.\n"
            "Создайте файл .env в корне проекта и добавьте строку:\n"
            "OPENAI_API_KEY=ваш_реальный_ключ"
        )
    return api_key


def call_hello_llm(client: OpenAI, temperature: float = 0.0) -> None:
    system_prompt = (
        "You are a helpful assistant who explains concepts clearly to a beginner Python developer."
    )
    user_prompt = "Объясни простыми словами, что делает параметр temperature в языковых моделях."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt},],
        )
    except AuthenticationError:
        print("Ошибка аутентификации: проверьте свой API-ключ (OPENAI_API_KEY).")
        return
    except APIError as e:
        print(f"Ошибка API: {e}")
        return
    except Exception as e:
        print(f"Неизвестная ошибка при вызове API: {type(e).__name__}: {e}")
        return

    choice = response.choices[0]
    message = choice.message
    finish_reason = choice.finish_reason

    print("=" * 80)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] temperature={temperature}")
    print("-" * 80)
    print("Ответ модели:")
    print(message.content)
    print("-" * 80)
    print(f"finish_reason: {finish_reason}")

    usage = getattr(response, "usage", None)
    if usage is not None:
        print(
            f"Токены: prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}"
        )
    else:
        print("Статистика по токенам недоступна в ответе.")


def run_temperature_experiment(client: OpenAI) -> None:
    print("\n=== Эксперимент с temperature = 0.0 (3 запуска) ===\n")
    for i in range(1, 4):
        print(f"\n--- Запуск {i} (temperature=0.0) ---")
        call_hello_llm(client, temperature=0.0)

    print("\n=== Эксперимент с temperature = 1.0 (3 запуска) ===\n")
    for i in range(1, 4):
        print(f"\n--- Запуск {i} (temperature=1.0) ---")
        call_hello_llm(client, temperature=1.0)


def main() -> None:
    api_key = load_api_key()

    client = OpenAI(api_key=api_key)

    print("=== Один тестовый вызов модели ===\n")
    call_hello_llm(client, temperature=0.5)

    print("\n\n=== Запускаем эксперимент с температурой ===")
    run_temperature_experiment(client)


if __name__ == "__main__":
    main()
