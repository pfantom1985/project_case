import os
import json
import time
from typing import List, Dict
from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APIError,
)
from token_utils import count_tokens, estimate_cost, check_context_fit

class ChatSession:

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.messages: List[Dict[str, str]] = []
        self.session_cost = 0.0


    def add_system_prompt(self, system_prompt: str) -> None:
        self.messages = [{"role": "system", "content": system_prompt}]
        print(f"Системная подсказка: {system_prompt[:60]}...")


    def check_context_warning(self) -> bool:
        context = check_context_fit(self.messages, self.model)
        usage_percent = context["input_tokens"] / context["context_window"] * 100

        if usage_percent > 80:
            print(
                f"Контекст: {usage_percent:.1f}% "
                f"({context['input_tokens']:,}/{context['context_window']:,})"
            )
            if not context["fits"]:
                print("История слишком большая!")
            return False
        return True


    def send_message(self, user_input: str, retry_count: int = 0) -> None:
        if not user_input.strip():
            return

        self.messages.append({"role": "user", "content": user_input})

        if not self.check_context_warning():
            return

        print("\nМодель отвечает...")

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=True,
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content

            print()

            self.messages.append({"role": "assistant", "content": full_response})

            prompt_tokens = count_tokens(
                json.dumps(self.messages[:-1], ensure_ascii=False),
                self.model,
            )
            completion_tokens = count_tokens(full_response, self.model)
            cost = estimate_cost(prompt_tokens, completion_tokens, self.model)
            self.session_cost += cost["total_cost"]

            print(f"\nВход: {prompt_tokens:,} (${cost['input_cost']:.6f})")
            print(f"   Выход: {completion_tokens:,} (${cost['output_cost']:.6f})")
            print(f"   Сессия: ${self.session_cost:.6f}")

        except AuthenticationError:
            print("\nПроверьте свой API-ключ в .env")
        except RateLimitError as e:
            if retry_count >= 3:
                print(f"\nRateLimitError (3/3): {e}")
                print("   Квота исчерпана. Чат остановлен.")
                return
            print(f"\nПопытка {retry_count + 1}/3: {e}")
            time.sleep(2 ** retry_count)  # 2, 4, 8 сек
            self.send_message(user_input, retry_count + 1)
        except APIConnectionError:
            print("\nПроверьте подключение к интернету")
        except APIError as e:
            print(f"\nAPI: {e}")
        except Exception as e:
            print(f"\nОшибка: {type(e).__name__}: {e}")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("OPENAI_API_KEY не найден в .env")
        return

    client = OpenAI(api_key=api_key)

    system_prompt = input(
        "Системная подсказка (Enter=Python tutor): "
    ).strip()
    if not system_prompt:
        system_prompt = "You are a helpful Python programming assistant."

    chat = ChatSession(client)
    chat.add_system_prompt(system_prompt)

    print("\nЧат запущен!")
    print("  'exit'  — выход\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
            if user_input.lower() in ["exit", "quit", "выход"]:
                print(f"\nИтого: ${chat.session_cost:.6f}")
                break
            chat.send_message(user_input)
        except KeyboardInterrupt:
            print("\nПрервано")
            break


if __name__ == "__main__":
    main()
