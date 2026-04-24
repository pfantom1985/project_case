import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIError
from pydantic import ValidationError

from ticket_schema import TicketStructuredOutput


INPUT_FILE = "data/tickets.jsonl"
OUTPUT_FILE = "results/structured_outputs.jsonl"
MODEL_NAME = "gpt-4o-2024-08-06"


def load_tickets(path: str) -> list[dict]:
    tickets = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tickets.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {path} on line {line_number}: {line}")
                raise e
    return tickets


def extract_structured_ticket(client: OpenAI, message: str) -> TicketStructuredOutput:
    system_prompt = """You are a support ticket triage assistant.

Convert the customer message into a structured support ticket object.

Rules:
- Choose exactly one category from the schema.
- Set priority based on urgency and impact.
- Set needs_human=true if the case involves refund disputes, duplicate charges, personal review requests, complaints requiring escalation, or urgent business-critical incidents.
- Detect reply_language from the user's message.
- draft_reply must be short, polite, and written in the same language as the user message.
- Do not add any extra fields.
"""

    try:
        response = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            response_format=TicketStructuredOutput,
            temperature=0,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raise ValueError("Model returned no parsed structured output.")

        return parsed

    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}") from e

    except RateLimitError as e:
        error_text = str(e)
        if "insufficient_quota" in error_text:
            raise RuntimeError(
                "OpenAI API quota exceeded. Check billing, usage limits, and project settings."
            ) from e
        raise RuntimeError("OpenAI API rate limit exceeded. Try again later.") from e

    except APIConnectionError as e:
        raise RuntimeError("Failed to connect to OpenAI API. Check your internet connection.") from e

    except APIError as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e


def save_results(results: list[dict], path: str) -> None:
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("OPENAI_API_KEY not found in .env")
        return

    client = OpenAI(api_key=api_key)

    try:
        tickets = load_tickets(INPUT_FILE)
    except Exception as e:
        print(f"Failed to load tickets: {e}")
        return

    results = []

    for ticket in tickets:
        case_id = ticket.get("case_id", "unknown")
        message = ticket.get("message", "")

        try:
            structured = extract_structured_ticket(client, message)
            result_row = {
                "case_id": case_id,
                "message": message,
                "structured_output": structured.model_dump(),
            }
            results.append(result_row)
            print(f"[OK] {case_id}: {structured.category}, {structured.priority}, needs_human={structured.needs_human}")

        except Exception as e:
            print(f"[ERROR] {case_id}: {e}")
            result_row = {
                "case_id": case_id,
                "message": message,
                "error": str(e),
            }
            results.append(result_row)

    save_results(results, OUTPUT_FILE)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()