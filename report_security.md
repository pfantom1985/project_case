## 1. Scenario

This test compares a naive baseline and a hardened variant under prompt injection pressure. The goal is to show that a safe system prompt alone is not enough, and that layered defenses reduce but do not eliminate risk. The evaluation covers direct injection, prompt leakage attempts, out-of-scope unsafe requests, and indirect attacks hidden in external text. This setup reflects common prompt-injection defense guidance that recommends layered controls, input separation, and output monitoring rather than relying on a single prompt boundary [web:209][web:242][web:247].

## 2. Attack cases

The dataset contains six attack cases and each is executed in two modes, producing 12 evaluated rows in total. Case types include a direct injection (`ignore previous instructions`), a system prompt leak attempt, an out-of-scope unsafe request, two indirect attacks embedded in external text, and one benign control case. This mix is enough to compare detection, blocking, and leakage behavior across both modes. It also makes it possible to observe both prevented attacks and residual failures in a single compact run [web:209][web:226][web:228].

## 3. Baseline behavior

The baseline is intentionally naive and often follows malicious instructions or echoes unsafe content. It is the weakest configuration because it does not reliably separate user input from external text and does not consistently block out-of-scope or prompt-leak requests. In this run, the baseline is the main source of unsafe exposure and prompt leakage. That is expected: without additional controls, direct injection and indirect manipulation remain easy to exploit [web:209][web:242][web:252].

## 4. Hardened behavior

The hardened version adds a scope guard, separates user input from external text, and applies basic rejection logic for suspicious content. In this run, it helped on four rows by blocking or preventing unsafe behavior, and it also kept the benign control case safe. The biggest improvement is that obvious attempts to reveal hidden instructions or force unsafe actions are no longer answered directly. This matches layered-defense guidance: isolate untrusted text, minimize trust, and monitor the output for leakage or unsafe behavior [web:242][web:247][web:253].

## 5. What exactly the protection measures improved

The hardened mode improved three concrete things. First, it reduced prompt leakage on direct attacks by refusing to reveal hidden instructions. Second, it blocked out-of-scope unsafe requests instead of answering them. Third, it treated external text as data rather than instructions, which prevented at least part of the indirect injection risk. In practice, this shows that basic prompt hardening can lower attack success, but only as one layer in a broader control stack [web:209][web:247][web:250].

## 6. Residual risks

The hardened setup still does not guarantee safety. Some risky cases were still not fully neutralized, especially where the attack was blended into normal-looking external text or where the model could still produce a partially unsafe or overly permissive answer. This is the main residual risk: simple heuristics catch obvious attacks, but paraphrased or semantically disguised injections can bypass them. The result is consistent with the broader security literature, which warns that prompt-level defenses slow attackers down but do not eliminate the problem [web:209][web:242][web:252].

## 7. Recommendation

Use hardened mode as the minimum baseline, but do not treat it as a complete security solution. Add a separate input classifier, stricter tool permissions, output validation, and human review for risky cases. For production, the best posture is layered defense: separate untrusted text, block suspicious requests early, and verify the output before acting on it. The experiment shows why one prompt is not enough and why defense-in-depth is required [web:209][web:247][web:253].