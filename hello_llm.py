import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, APIError

def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY не найден!\n"
            "Создайте .env:\nOPENAI_API_KEY=ваш_ключ"
        )
    return api_key

def call_hello_llm(client: OpenAI, temperature: float = 0.0) -> None:
    system_prompt = (
        "You are a helpful assistant who explains concepts clearly "
        "to a beginner Python developer."
    )
    user_prompt = "Объясни простыми словами, что делает параметр temperature в языковых моделях."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except AuthenticationError:
        print("Ошибка аутентификации: проверьте API-ключ.")
        return
    except APIError as e:
        print(f"Ошибка API: {e}")
        return
    except Exception as e:
        print(f"Ошибка: {type(e).__name__}: {e}")
        return
    choice = response.choices[0]
    print("=" * 80)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] temp={temperature}")
    print("Ответ модели:")
    print(choice.message.content)
    print(f"finish_reason: {choice.finish_reason}")
    usage = getattr(response, "usage", None)
    if usage:
        print(f"Токены: {usage.prompt_tokens} + {usage.completion_tokens} = {usage.total_tokens}")
    else:
        print("Статистика токенов недоступна.")

def run_temperature_experiment(client: OpenAI) -> None:
    print("\n=== temperature=0.0 (3x) ===\n")
    for i in range(1, 4):
        print(f"--- #{i} ---")
        call_hello_llm(client, 0.0)
    print("\n=== temperature=1.0 (3x) ===\n")
    for i in range(1, 4):
        print(f"--- #{i} ---")
        call_hello_llm(client, 1.0)


def main() -> None:
    client = OpenAI(api_key=load_api_key())
    print("Тестовый вызов:\n")
    call_hello_llm(client, 0.5)
    print("\nЭксперимент temperature:")
    run_temperature_experiment(client)

if __name__ == "__main__":
    main()
