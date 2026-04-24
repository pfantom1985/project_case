import os
import json
import csv
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIError

PROMPTS = {
    "support_v1": "prompts/support_v1.txt",
    "support_v2": "prompts/support_v2.txt",
}

TEST_CASES_FILE = "data/test_cases.jsonl"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

EVAL_CSV = RESULTS_DIR / "prompt_eval.csv"
SUMMARY_CSV = RESULTS_DIR / "prompt_summary.csv"

MODEL_NAME = "gpt-4o-mini"


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_test_cases(path: str) -> list[dict]:
    cases = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Invalid JSON on line {line_number}: {line}")
                raise e
    return cases


def call_model(client: OpenAI, system_prompt: str, user_message: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    except RateLimitError as e:
        error_text = str(e)
        if "insufficient_quota" in error_text:
            raise RuntimeError(
                "OpenAI API quota exceeded. Check billing, usage limits, and project selection."
            ) from e
        raise RuntimeError("OpenAI API rate limit exceeded. Try again later.") from e

    except APIConnectionError as e:
        raise RuntimeError("Failed to connect to OpenAI API. Check your internet connection.") from e

    except APIError as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e


def is_answered(answer: str) -> bool:
    return bool(answer and answer.strip())


def is_refusal(answer: str) -> bool:
    answer_lower = answer.lower()
    refusal_markers = [
        "i can only help",
        "out of scope",
        "contact a human agent",
        "please contact a human agent",
        "i’m unable to help",
        "i cannot help",
        "i can't help",
    ]
    return any(marker in answer_lower for marker in refusal_markers)


def is_clarification(answer: str) -> bool:
    answer_lower = answer.lower()
    clarification_markers = [
        "could you clarify",
        "can you clarify",
        "please clarify",
        "which",
        "what exactly",
        "can you share",
        "please share",
    ]
    return any(marker in answer_lower for marker in clarification_markers) or "?" in answer


def is_format_ok(answer: str) -> bool:
    if not answer or not answer.strip():
        return False
    if len(answer) > 1200:
        return False
    return True


def evaluate_case(expected_behavior: str, answer: str) -> dict:
    answered = is_answered(answer)
    refused = is_refusal(answer)
    clarified = is_clarification(answer)
    format_ok = is_format_ok(answer)

    refused_correctly = expected_behavior == "refuse" and refused

    return {
        "answered": answered,
        "refused_correctly": refused_correctly,
        "format_ok": format_ok,
        "answer_length": len(answer),
        "detected_refusal": refused,
        "detected_clarification": clarified,
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "case_id",
        "prompt_version",
        "answered",
        "refused_correctly",
        "format_ok",
        "answer_length",
    ]

    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    separator = "-+-".join("-" * widths[h] for h in headers)

    print(header_line)
    print(separator)

    for row in rows:
        print(" | ".join(str(row[h]).ljust(widths[h]) for h in headers))


def save_eval_csv(rows: list[dict], path: Path) -> None:
    headers = [
        "case_id",
        "prompt_version",
        "in_scope",
        "expected_behavior",
        "answered",
        "refused_correctly",
        "format_ok",
        "answer_length",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row[h] for h in headers})


def build_summary(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        version = row["prompt_version"]
        grouped.setdefault(version, []).append(row)

    summary = []
    for version, items in grouped.items():
        total_cases = len(items)
        answered_count = sum(1 for x in items if x["answered"])
        correct_refusals = sum(1 for x in items if x["refused_correctly"])
        format_ok_count = sum(1 for x in items if x["format_ok"])
        avg_answer_length = round(
            sum(x["answer_length"] for x in items) / total_cases, 2
        )

        score = (correct_refusals * 3) + format_ok_count + answered_count
        summary.append(
            {
                "prompt_version": version,
                "total_cases": total_cases,
                "answered_count": answered_count,
                "correct_refusals": correct_refusals,
                "format_ok_count": format_ok_count,
                "avg_answer_length": avg_answer_length,
                "score": score,
            }
        )

    best_version = max(summary, key=lambda x: x["score"])["prompt_version"]

    for row in summary:
        row["recommended_for_next_step"] = row["prompt_version"] == best_version

    return summary


def save_summary_csv(summary_rows: list[dict], path: Path) -> None:
    headers = [
        "prompt_version",
        "total_cases",
        "answered_count",
        "correct_refusals",
        "format_ok_count",
        "avg_answer_length",
        "recommended_for_next_step",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({h: row[h] for h in headers})


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("OPENAI_API_KEY not found in .env")
        return

    client = OpenAI(api_key=api_key)

    test_cases = load_test_cases(TEST_CASES_FILE)
    prompts = {name: load_text(path) for name, path in PROMPTS.items()}

    all_rows = []

    for case in test_cases:
        for prompt_version, system_prompt in prompts.items():
            try:
                answer = call_model(client, system_prompt, case["user_message"])
            except RuntimeError as e:
                print(f"Error for {case['case_id']} / {prompt_version}: {e}")
                return

            metrics = evaluate_case(case["expected_behavior"], answer)

            row = {
                "case_id": case["case_id"],
                "prompt_version": prompt_version,
                "in_scope": case["in_scope"],
                "expected_behavior": case["expected_behavior"],
                "answered": metrics["answered"],
                "refused_correctly": metrics["refused_correctly"],
                "format_ok": metrics["format_ok"],
                "answer_length": metrics["answer_length"],
            }
            all_rows.append(row)

    print_table(all_rows)
    save_eval_csv(all_rows, EVAL_CSV)

    summary_rows = build_summary(all_rows)
    save_summary_csv(summary_rows, SUMMARY_CSV)

    print(f"\nSaved: {EVAL_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()