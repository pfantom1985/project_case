import os
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIError
from token_utils import count_tokens, estimate_cost, check_context_fit


class ChatSession:
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.messages: List[Dict[str, str]] = []
        self.session_cost = 0.0  # Накопительная стоимость сеанса

    def add_system_prompt(self, system_prompt: str) -> None:
        """Добавляет системную подсказку как первое сообщение."""
        self.messages = [{"role": "system", "content": system_prompt}]
        print(f"Системная подсказка установлена: {system_prompt[:60]}...")

    def check_context_warning(self) -> bool:
        """Проверяет контекст и предупреждает, если >80% занято."""
        context = check_context_fit(self.messages, self.model)
        usage_percent = context["input_tokens"] / context["context_window"] * 100

        if usage_percent > 80:
            print(
                f"Контекст заполнен на {usage_percent:.1f}% ({context['input_tokens']:,}/{context['context_window']:,})")
            if not context["fits"]:
                print("История слишком большая! Рассмотрите начало нового чата.")
            return False
        return True

    def send_message(self, user_input: str) -> None:
        """Отправляет сообщение пользователя и печатает потоковый ответ."""
        if not user_input.strip():
            return

        # DEBUG режим: тест без API
        if user_input.strip().lower() == "test":
            print("\nDEBUG: Имитация ответа без API...")
            full_response = "Это тестовый ответ. Все функции работают корректно!"
            print(full_response)

            # Добавляем в историю для статистики
            self.messages.append({"role": "user", "content": user_input})
            self.messages.append({"role": "assistant", "content": full_response})

            # Подсчёт для отладки
            prompt_tokens = count_tokens(json.dumps(self.messages[:-1]), self.model)
            completion_tokens = count_tokens(full_response, self.model)
            cost = estimate_cost(prompt_tokens, completion_tokens, self.model)
            self.session_cost += cost["total_cost"]

            print(f"\nDEBUG статистика: ${cost['total_cost']:.6f}")
            return

        # Добавляем сообщение пользователя
        self.messages.append({"role": "user", "content": user_input})

        # Проверяем контекст
        if not self.check_context_warning():
            return

        print("\n🤖 Модель печатает ответ...")

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=True,
            )

            # Собираем полный ответ
            full_response = ""
            prompt_tokens_before = count_tokens(json.dumps(self.messages[:-1]), self.model)

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content

            print()  # новая строка после ответа

            # Добавляем ответ модели в историю
            self.messages.append({"role": "assistant", "content": full_response})

            # Подсчёт токенов и стоимости
            completion_tokens = count_tokens(full_response, self.model)
            prompt_tokens = count_tokens(json.dumps(self.messages[:-1]), self.model)

            cost = estimate_cost(prompt_tokens, completion_tokens, self.model)
            self.session_cost += cost["total_cost"]

            # Вывод статистики
            print(f"\nСтатистика:")
            print(f"   Вход: {prompt_tokens:,} токенов (${cost['input_cost']:.6f})")
            print(f"   Выход: {completion_tokens:,} токенов (${cost['output_cost']:.6f})")
            print(f"   Сессия: ${self.session_cost:.6f} всего")

        except AuthenticationError:
            print("\nAuthenticationError: Проверьте свой API-ключ в .env")
        except RateLimitError as e:
            print(f"\nRateLimitError: {e}")
            print("   Автоматическая повторная попытка через 5 сек...")
            time.sleep(5)
            # Рекурсивно повторяем запрос (сообщение пользователя уже добавлено)
            self.send_message(user_input)
            return
        except APIConnectionError:
            print("\nAPIConnectionError: Проверьте подключение к интернету")
        except APIError as e:
            print(f"\nAPIError: {e}")
        except Exception as e:
            print(f"\nНеожиданная ошибка: {type(e).__name__}: {e}")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Не найден OPENAI_API_KEY в .env")
        return

    client = OpenAI(api_key=api_key)

    # Системная подсказка
    system_prompt = input("Введите системную подсказку (или Enter для 'You are helpful assistant'): ").strip()
    if not system_prompt:
        system_prompt = "You are a helpful Python programming assistant."

    chat = ChatSession(client)
    chat.add_system_prompt(system_prompt)

    print("\nЧат запущен!")
    print("   • 'test' — тест без API")
    print("   • 'exit'/'quit'/'выход' — выход\n")

    while True:
        try:
            user_input = input("Вы: ").strip()

            if user_input.lower() in ["exit", "quit", "выход"]:
                print(f"\nСессия завершена. Итого потрачено: ${chat.session_cost:.6f}")
                break

            chat.send_message(user_input)

        except KeyboardInterrupt:
            print("\n\nЧат прерван пользователем.")
            break


if __name__ == "__main__":
    main()
