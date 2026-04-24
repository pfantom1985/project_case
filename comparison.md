# Сравнение подходов модуля 2

| Задание | Baseline | Hardened | Улучшение |
|---------|----------|----------|-----------|
| P.1 Prompt eval | answered=True, refused_correctly=False | answered=True, refused_correctly=True на case_07 | Лучше guardrail |
| P.2 Structured | Free text | Pydantic schema | Автоматическая маршрутизация |
| P.3 Security | Unsafe responses | 4/6 blocked | Многоуровневая защита |