# Phase 2 — Cross-Model Replication Plan (LOCKED)

**Status:** strategy locked; implementation not started.
**Scope:** cross-model replication only. No benchmark, dataset, scoring, prompt
(v0), U1 subset, U1 decision rule, or A1-ceiling changes.
**Motivation:** the highest-priority weakness in the project is that every
empirical claim currently rests on a single model (DeepSeek, n=1). Phase 2
removes that weakness by replicating Experiment 0 (calibration / confidently-
wrong) and U1 (self-consistency as a safety signal) across additional
off-the-shelf LLMs through the *identical* frozen instrument.

---

## 1. Invariants (frozen)

The following do **not** change in Phase 2:

- Benchmark core: `run.py`, `scoring.py`, the BFS oracle.
- Prompt **v0**; dataset `data/eval_dataset.json` (1200 samples).
- U1 subset (seed 20240608, n=300, 100/diff) and the locked U1 FOR/AGAINST
  decision rule (5-criterion conjunction).
- Authority ceiling **A1** and the U1 = AGAINST result for DeepSeek.

The only degree of freedom Phase 2 uses is **adding a model adapter** per
provider — exactly as the GLM adapter was added previously. An adapter is not a
benchmark/dataset/scoring change; it runs a new subject through the frozen
instrument.

## 2. Two experiments per model (both frozen)

| Experiment | Decoding | Calls/model | Produces |
| --- | --- | --- | --- |
| Exp 0 (calibration / confidently-wrong) | temp-0, 1200 samples | 1,200 | `results/<model>.json` |
| U1 (self-consistency) | temp-0.7, n=300, k=10 | 3,000 | `results/<model>_u1.json` |

## 3. Pre-registration (decide before running)

- **ED replicates on a model** iff confidently-wrong rate ≥ 25% **and** ECE ≥
  0.20 (DeepSeek reference: 42.5% / 0.425).
- **Self-consistency fails on a model** iff the existing locked U1 FOR/AGAINST
  rule returns **AGAINST** for that model (reused verbatim).
- **Malformed-handling (locked):** a response the frozen v0 parser cannot parse
  is scored as a **failure** (no prompt tweaking to rescue it). **Exclusion
  threshold:** if Exp 0 malformed rate > **20%**, the model is reported as
  *protocol-incompatible under frozen v0*, not mixed into the ED/U1 comparison.
- **Reporting unit:** per-model first; pooled/distributional claims only at n≥3.

---

## 4. Repository review (line-level findings)

### 4.1 Adapters that exist

| Adapter | Location | Status |
| --- | --- | --- |
| `GreedyAgent` | `agents.py` (both branches) | baseline, complete |
| `DeepSeekAgent` | `agents.py` (both branches) | complete; Exp 0 + U1 done |
| `GLMAgent` | `agents.py` — **feature branch `claude/grid-navigation-status-review-0KN28` (PR #4) only** | complete, production-hardened |
| Kimi / MiMo | — | do not exist |

`GLMAgent` is fully built: OpenAI-compatible `/chat/completions` via stdlib
`urllib`, credential stripping (trailing-newline header fix), transient
5xx + connection-reset retry with exponential backoff, configurable `base_url`
(Zhipu BigModel CN host and international Z.ai host), default model
`glm-4-plus`. The feature branch already wires it into the Exp 0 runner:
`evaluate.py` has `--agent glm`, `make_agent` returns `GLMAgent(...)`, and
`glm-smoke.yml` / `glm-benchmark.yml` workflows exist.

### 4.2 Decisive portability fact

DeepSeek and GLM are **byte-for-byte OpenAI-compatible** — identical request
payload, identical `body["choices"][0]["message"]["content"]` extraction. Any
provider with an OpenAI-compatible chat endpoint is therefore a *parameterized
clone* of `GLMAgent` (different base_url/key/model), not new architecture.

### 4.3 Two integration gaps (both small)

1. **Branch split (the real friction).** U1 (`u1_selfconsistency.py`) lives
   **only on the default branch**; `GLMAgent` lives **only on the feature
   branch (PR #4)**. Neither branch can currently run GLM-U1. Consolidation =
   bring `GLMAgent` + `evaluate.py` GLM wiring + the two GLM workflows onto the
   default branch (a merge/cherry-pick, not new code).
2. **U1 is hardwired to DeepSeek.** `u1_selfconsistency.py` imports only
   `DeepSeekAgent` (line 50), instantiates it directly (line 370), and stamps
   `"provider": "deepseek"` (line 550). Cross-model U1 needs a small
   **additive** agent-selector. Exp 0 (`evaluate.py`) already has its selector,
   so GLM-Exp 0 is ready today on the feature branch.

### 4.4 Freeze-tension decision point (requires explicit sign-off)

Adding the agent-selector to the "frozen" `u1_selfconsistency.py`. Two options
preserve the freeze:

- **(a) Parameterize U1 additively** with `--agent` defaulting to `deepseek`;
  frozen behavior and the committed DeepSeek result are byte-identical when the
  flag is unused. *(Recommended for simplicity.)*
- **(b) Thin sibling wrapper** that imports U1's functions and swaps only the
  agent, leaving the file untouched.

No change will be made to this file without explicit approval of (a) or (b).

---

## 5. Execution order and per-model assessment

DeepSeek is done. Actionable order: **GLM → Kimi → MiMo**, each behind a
dry-run malformed gate.

| Model | Eng. effort | API cost (Exp 0 + U1) | Runtime | Scientific value | Risk |
| --- | --- | --- | --- | --- | --- |
| **DeepSeek** (n=1) | none (done) | sunk | done | establishes ED + U1 AGAINST | none |
| **GLM** (→ n=2) | **LOW** — adapter/wiring/workflows exist; consolidate + additive U1 selector + dry-run | ~$0.3–2.8 (Flash–Plus) | ~1.5–2 h | **HIGHEST per effort** — kills n=1 on both ED and U1 | LOW (format divergence; gate catches) |
| **Kimi** (→ n=3) | LOW–MEDIUM — clone GLMAgent, wire `--agent kimi`, key, dry-run | ~$5 | ~1.5–2 h (+rate limits) | HIGH — distributional ED claim + cross-model U1 | LOW–MEDIUM |
| **MiMo** (→ n=4) | MEDIUM–HIGH — adapter trivial, **serving path** is the work | <$1.5 | ~1.5–2 h + unbounded setup | MEDIUM — different family/scale; diminishing returns | **HIGH** — no guaranteed hosted OpenAI-compatible API |

**Cost across the whole panel is negligible (< ~$15–20 total).** The binding
resources are integration effort and the MiMo serving path, not dollars.
Runtime anchored to observed DeepSeek throughput (U1: 3000 calls ≈ 71 min ⇒
~1.4 s/call); slow/rate-limited models may be 2–3× and U1 may exceed the
180-min CI timeout (chunk/resume).

### 5.1 MiMo serving-path options (gate before committing)

- **OpenRouter** route (if it carries MiMo) → OpenAI-compatible → adapter
  trivial; depends on availability.
- **Self-hosted vLLM** behind an OpenAI-compatible shim → adapter trivial, but
  must stand up + reach a GPU endpoint from CI (egress/secret/infra).
- If neither is clean → report MiMo as protocol-incompatible / **deferred**.

---

## 6. Recommended stopping point (12-month research-value horizon)

**Target n=3 (DeepSeek + GLM + Kimi). Treat n=4 (MiMo) as a stretch, gated on a
serving-path spike.**

- **n=2 (GLM)** — mandatory minimum; removes the n=1 objection on both claims at
  near-zero engineering cost (adapter already exists). Non-negotiable.
- **n=3 (Kimi)** — the value-per-effort sweet spot; turns "replicated once" into
  a cross-model *pattern* and supports the distributional ED claim. Recommended
  target.
- **n=4 (MiMo)** — real but diminishing external-validity value; the only HIGH
  integration risk. Pursue only if (a) a serving path is confirmed cheaply and
  (b) explicitly aiming at publication. Must not block n=3.

### 6.1 Replication levels mapped to thesis ambition

- **Minimum to remove n=1:** DeepSeek + GLM (n=2), Exp 0 **and** U1.
- **Strong undergraduate thesis:** n=3 (DeepSeek + GLM + Kimi); Exp 0 on all
  three, U1 on at least two.
- **Publication-oriented:** n=4 full panel with the pre-registered exclusion
  rule and per-model + pooled reporting (paired later, separate phases, with the
  elicitation-robustness arm and closed-loop validation — both out of Phase 2
  scope).

### 6.2 Decision gates (off-ramps)

GLM (n=2) → inspect ED + U1 replication → Kimi (n=3) → broad Exp 0 panel →
decide MiMo / cross-model U1 based on whether results converge or diverge. A
*divergent* result (e.g., a model with varied, calibrated confidence) is a
**finding**, not a failure — it localizes ED to specific training regimes.

---

## 7. First implementation action (when authorized)

1. Consolidate `GLMAgent` + `evaluate.py` GLM wiring + GLM workflows from PR #4
   onto the default branch so GLM and U1 coexist.
2. Resolve the §4.4 U1 agent-selector decision (a or b).
3. Run the GLM dry-run gate (10–20 samples) → go/no-go on malformed rate.
4. GLM Exp 0 → GLM U1 → apply locked rule and pre-registration thresholds.

*Out of Phase 2 scope (future, separate phases):* confidence-elicitation
robustness arm (prompt v1), closed-loop/trajectory validation of the A1
architecture, and U2 (alternative uncertainty signals — postponed in favor of
breadth).
