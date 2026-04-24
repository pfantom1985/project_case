# Заметки по структурированному выводу

**Зачем Pydantic + structured outputs:**
- `response_format=TicketStructuredOutput` гарантирует валидный JSON
- `client.beta.chat.completions.parse()` сразу парсит в типизированный объект
- `model_validate()` проверяет схему на выходе

**Пример в коде:**
```python
response = client.beta.chat.completions.parse(
    ..., response_format=TicketStructuredOutput
)
ticket = response.choices.message.parsed
if ticket.needs_human:
    route_to_human(ticket)
```

**Разница с "красивым текстом":**
- Free text: "Это биллинг, срочно, нужен человек"
- Structured: `{"category": "billing", "priority": "urgent", "needs_human": true}`