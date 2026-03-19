import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ OPENAI_API_KEY не найден в .env")
    exit(1)

client = OpenAI(api_key=api_key)

try:
    # Лёгкий запрос — список моделей (дешево!)
    models = client.models.list()
    print("✅ API ключ ВАЛИДНЫЙ!")
    print(f"   Доступно моделей: {len(models.data)}")
    print(f"   Первая модель: {models.data[0].id}")
except Exception as e:
    print(f"❌ ОШИБКА API: {e}")