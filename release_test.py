import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, AuthenticationError, RateLimitError
from langfuse import Langfuse
from token_utils import count_tokens, estimate_cost


load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
)

MODEL = "gpt-4o-mini"


def load_questions(filename: str = "test_questions.txt") -> list[str]:
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_release_test(release: str, prompt_version: int, max_tokens: int) -> None:
    questions = load_questions()
    prompt_obj = langfuse.get_prompt("helper-prompt", version=prompt_version)
    prompt_text = str(prompt_obj.compile())

    print(f"Run release {release}...")

    for question in questions:
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": question},
        ]

        try:
            trace = langfuse.trace(
                name="release-test",
                user_id="staging-user",
                input={
                    "question": question,
                    "release": release,
                    "prompt_version": prompt_version,
                },
                output=None,
                metadata={
                    "environment": "staging",
                    "prompt_version": f"v{prompt_version}",
                    "model": MODEL,
                    "release": release,
                    "max_tokens": max_tokens,
                },
                tags=["staging", "regression-test", release],
            )

            prompt_span = trace.span(
                name="prompt_assembly",
                input={
                    "messages": messages,
                    "prompt_name": "helper-prompt",
                    "prompt_version": prompt_version,
                },
                metadata={
                    "release": release,
                },
            )
            prompt_span.end(
                output={
                    "assembled_messages": messages
                }
            )

            started_at = time.time()

            response = openai_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )

            latency = time.time() - started_at
            answer = response.choices[0].message.content or ""

            input_tokens = count_tokens(
                json.dumps(messages, ensure_ascii=False),
                MODEL,
            )
            output_tokens = count_tokens(answer, MODEL)
            total_tokens = input_tokens + output_tokens
            cost = estimate_cost(input_tokens, output_tokens, MODEL)

            generation = trace.generation(
                name="llm_call",
                model=MODEL,
                input=messages,
                output=answer,
                prompt=prompt_obj,
                metadata={
                    "release": release,
                    "environment": "staging",
                    "latency_seconds": round(latency, 3),
                },
                model_parameters={
                    "temperature": 0.7,
                    "max_tokens": max_tokens,
                },
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            )
            generation.end()

            trace.update(
                output={
                    "answer": answer,
                    "release": release,
                },
                metadata={
                    "environment": "staging",
                    "prompt_version": f"v{prompt_version}",
                    "model": MODEL,
                    "release": release,
                    "max_tokens": max_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "total_cost": cost["total_cost"],
                    "latency_seconds": round(latency, 3),
                },
                tags=["staging", "regression-test", release],
            )

            langfuse.flush()

            print(
                f"[{release}] {question[:40]}... | "
                f"in={input_tokens} out={output_tokens} "
                f"cost=${cost['total_cost']:.6f} latency={latency:.2f}s"
            )

        except AuthenticationError:
            print("OpenAI auth error: проверь OPENAI_API_KEY")
        except RateLimitError as e:
            print(f"Rate limit: {e}")
        except APIConnectionError:
            print("OpenAI connection error")
        except APIError as e:
            print(f"OpenAI API error: {e}")
        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    run_release_test(release="v1.0.0", prompt_version=1, max_tokens=200)
    run_release_test(release="v1.1.0", prompt_version=2, max_tokens=500)
    print("Done.")