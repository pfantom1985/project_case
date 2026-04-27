# Engineering Report: Release Regression

## 1. Compared releases
- **Release v1.0.0**: Base prompt, gpt-4o-mini, max_tokens=200.
- **Release v1.1.0**: Updated formatting instructions, gpt-4o-mini, max_tokens=500.

## 2. Symptom
After deploying v1.1.0, the average token output and cost increased by ~45%, leading to higher latency.

## 3. Metrics before the change (v1.0.0)
- Avg Input Tokens: 45
- Avg Output Tokens: 82
- Avg Cost: $0.00008

## 4. Metrics after the change (v1.1.0)
- Avg Input Tokens: 48
- Avg Output Tokens: 185
- Avg Cost: $0.00018

## 5. Confirmed cause
The new instruction "Always provide 3 examples..." in v1.1.0 forced the model to generate significantly longer responses, increasing output token usage.

## 6. Which Langfuse data confirmed this
Trace comparison in Langfuse showed a consistent spike in `output_tokens` and `total_cost` for traces with `release=v1.1.0`.

## 7. Recommendation
Roll back the formatting requirements or introduce token-based constraints in the system prompt to limit verbosity.