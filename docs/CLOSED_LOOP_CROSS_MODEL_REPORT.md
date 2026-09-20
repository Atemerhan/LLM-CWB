# CLOSED_LOOP_CROSS_MODEL_REPORT.md — 闭环真实 LLM 臂（跨模型）

Trajectory-level results of the 档A closed-loop simulation with a **real LLM** as
the L3 policy (`run_embodied_llm.py`), per authority arm. Same frozen grids as
episodes (n=100); decisions live. Run on GitHub Actions (runs 27818597172 /
…601404 / …606119 / …610823); numbers from the run logs.

## 1. Per-arm metrics (n=100 episodes)

| Model | arm | success | SPL | collision | veto | llm_calls |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek `deepseek-chat` | A1 | 0.570 | 0.557 | **0.000** | 0.129 | 262 |
| | ungated | 0.370 | 0.362 | **0.310** | — | |
| | planner | 1.000 | 1.000 | 0.000 | — | |
| GLM `glm-4-flash` | A1 | 0.380 | 0.355 | **0.000** | 0.488 | 388 |
| | ungated | 0.040 | 0.040 | **0.960** | — | |
| | planner | 1.000 | 1.000 | 0.000 | — | |
| Kimi `moonshot-v1-8k` | A1 | 0.340 | 0.328 | **0.000** | 0.465 | 370 |
| | ungated | 0.050 | 0.050 | **0.930** | — | |
| | planner | 1.000 | 1.000 | 0.000 | — | |
| **GLM `glm-5.1`** (upgrade) | A1 | 1.000 | 1.000 | **0.000** | 0.000 | 208 |
| | ungated | 1.000 | 1.000 | **0.000** | — | |
| | planner | 1.000 | 1.000 | 0.000 | — | |

malformed = 0 for all. Baseline models pinned to the Exp 0 / U1 ids (deepseek
-chat / glm-4-flash / moonshot-v1-8k); glm-5.1 is an additional upgrade-model run.

## 2. Headline
- **A1 holds the collision rate at exactly 0.000 for every model** — the safety
  -by-construction result, now with *real LLMs* in a closed loop (not just the
  greedy/random reference policies of the kinematic n=1200 run).
- **The three baseline models confidently crash when ungated**: deepseek 31.0%,
  Kimi 93.0%, GLM-4-flash 96.0% of episodes end in collision. The per-step unsafe
  behaviour compounds over the trajectory exactly as the S3 projection predicts —
  and A1 removes all of it, paying only SPL (deepseek 0.557, GLM 0.355, Kimi
  0.328) and veto-load (0.13 / 0.49 / 0.47, echoing the single-step unsafe rates).
- **The upgrade model (glm-5.1) is so capable it proposes zero unsafe moves**:
  ungated collision 0, A1 veto 0, SPL 1.000 — on this easy fully-observed grid it
  matches the A\* planner, so A1 is a no-op safety wrapper (zero overhead). An
  honest, informative contrast, not a contradiction (see §3).

## 3. Reading (honest framing for the thesis)
1. **The architecture's value is proven on the models that need it.** A1 converts
   a 31–96% real-LLM collision rate into 0, at an efficiency cost — the
   constructive "safety-for-efficiency" trade-off, with real models.
2. **Capability is rising fast but does not retire the safety question.** glm-5.1
   solving this benchmark perfectly says the *fully-observed 8×8 grid is now
   trivial for a frontier model* — not that frontier models are safe in general.
   The grid is a measurement instrument; the safety question reopens in the
   harder / partial-observable / semantic regimes (where an LLM is actually needed
   and A\* no longer suffices). n=100 perfection is not a deployment guarantee.
3. **A1 is model-agnostic and zero-cost when unnecessary.** For glm-5.1 the
   monitor never fired (veto 0), so bounding authority costs nothing when the
   model is already safe — and still catches the models that aren't. That is the
   point of safety-by-construction: it does not depend on *which* model is behind it.

## 4. Thesis slot-in
Fills the in-progress real-LLM rows of 中文论文 §4.7 and the closed-loop
contribution in §6. Pair with the kinematic n=1200 run (reference policies) and
the AirSim physics-engine run (`EMBODIED_AIRSIM_RESULTS.md`) as three fidelity
levels of the same constructive result.
