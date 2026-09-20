# ARCHITECTURE — Safety-Bounded Hierarchical LLM Decision Architecture for UAV Navigation

**Status:** FROZEN (documentation only). No code, metrics, architecture blocks, or
research dimensions are added beyond what is recorded here. Companion document:
`docs/SPEC_v2.md` (evaluation framework + metric-to-layer mapping).

**Thesis:** *Research on LLM-Based Decision-Making Methods for UAV Navigation —
A Safety-Bounded Hierarchical Architecture with Confidence-Gated Runtime Assurance.*

---

## 1. Positioning

The LLM is **not** a replacement for a classical planner (A\*/D\*), which already
solves the frozen grid optimally, safely, deterministically, and in microseconds.
The LLM is positioned as a **high-level / supervisory decision element** inside a
**Simplex / Run-Time Assurance (RTA)** hierarchy: an unverified, high-capability
controller *proposes*; a verified safety controller *disposes*. The LLM is granted
**bounded authority** that scales with its measured trustworthiness (see §4–§5).

**Core invariant:** the LLM never commands an actuator directly. Its best case is to
propose a sub-goal that the classical planner executes and the safety monitor
verifies. Therefore an LLM error can cause a *veto + fallback* (a performance loss),
**never a safety violation** (a crash). Safety is guaranteed by construction at L1.

---

## 2. Layered architecture (L0–L3)

| Layer | Element | Role | Cadence | Guarantee |
| --- | --- | --- | --- | --- |
| **L3** | **LLM** | Strategic/semantic decisions A\* cannot make: interpret an ambiguous / natural-language mission, select sub-goal / mode, decide *when* to replan, reason about contingencies. Emits `(decision dₜ, uncertainty uₜ)`. | slow, ~0.1–1 Hz | **None** (untrusted, high-performance) |
| **L2** | **Authority arbiter / fallback** | Decide whether to *accept, defer, or override* the LLM, using uncertainty + safety + deadline gates. | per decision | bounded authority; deterministic fallback |
| **L2** | **Classical planner** | Geometric path to the (LLM- or default-) sub-goal; the default controller whenever the LLM is rejected/deferred. | mid | optimal/complete given map + goal |
| **L1** | **Safety monitor / verified executor** | Verify *every* command — whatever its source — for collision-freedom, geofence, kinematic & energy limits. | every command | **hard safety (by construction)** |
| **L0** | **Flight controller** | Stabilize and track the verified command (e.g. PX4 / ArduPilot inner loop). | fast, ~50–1000 Hz | inner-loop stability |

---

## 3. Signal flow

```
        Mission / semantic goal (NL, constraints, priorities)
                              │
   ┌──────────────────────────▼──────────────────────────────┐
   │  L3  HIGH-LEVEL DECISION LAYER  ── LLM ──   (~0.1–1 Hz)   │
   │  • interpret goal / constraints                          │
   │  • select sub-goal, mode, or strategy                    │
   │  • decide WHEN to replan / handle contingencies          │
   │  outputs:  decision dₜ  +  uncertainty uₜ                 │
   └──────────────────────────┬──────────────────────────────┘
                  proposed (dₜ, uₜ)│
   ┌──────────────────────────▼──────────────────────────────┐
   │  L2  AUTHORITY ARBITER / FALLBACK LOGIC  (RTA decision)  │
   │  gates: deadline B · parse-valid · Safe(d|x) · u ≤ τ     │
   └───────┬───────────────────────────────────────┬──────────┘
   accept  │ (sub-goal)                      reject/defer/timeout
           │                                          │
   ┌───────▼────────────┐                  ┌──────────▼─────────────┐
   │ use LLM sub-goal   │                  │  L2  CLASSICAL PLANNER │
   │ as planner target  │                  │  A*/D*  (+ greedy      │
   └───────┬────────────┘                  │  reactive fallback)    │
           │                               └──────────┬─────────────┘
           └───────────────┬──────────────────────────┘
                geometric path / waypoint
   ┌──────────────────────▼──────────────────────────────────┐
   │  L1  SAFETY MONITOR / VERIFIED EXECUTOR  (HARD guarantee)│
   │  collision-check · geofence · kinematic & energy limits  │
   │  every command — whatever its source — passes here       │
   └──────────────────────┬──────────────────────────────────┘
                verified command
   ┌──────────────────────▼──────────────┐
   │  L0  FLIGHT CONTROLLER (inner loop)  │   (~50–1000 Hz)
   └──────────────────────┬──────────────┘
                          ▼
                 Vehicle / Environment  ──► state feedback (up to L3, L2, L1)
```

---

## 4. Authority ladder

The LLM's privilege scales with measured trust (parameterized by the metrics in
`docs/SPEC_v2.md`). The benchmark determines the achievable level.

| Level | Name | LLM privilege | When applied |
| --- | --- | --- | --- |
| **A0** | Advisory | Output logged, never acted | Poor confidently-unsafe rate (S2) / poor AURC (S4) |
| **A1** | Select-from-safe | LLM picks among planner-generated **safe** options only | Moderate trust |
| **A2** | **Sub-goal proposal** *(target)* | LLM proposes sub-goals; planner + monitor execute/verify | Low unsafe-rate + usable uncertainty |
| **A3** | Direct waypoint | LLM emits waypoints (still monitor-checked at L1) | High trust only |

---

## 5. Fallback logic / trust-gated arbitration

For proposed decision `d`, uncertainty `u`, vehicle state `x`, deadline budget `B`,
safety predicate `Safe(d | x)` (evaluated by L1), and uncertainty threshold `τ`:

```
if   not received within B            →  FALLBACK(planner)      # real-time gate   (L1*/L2/L4 metrics)
elif response malformed / unparsable  →  FALLBACK(planner)      # reliability gate (R4)
elif not Safe(d | x)                  →  REJECT → FALLBACK      # safety gate      (S1/S2 caught here)
elif u > τ                            →  DEFER  → FALLBACK      # trust gate       (U1, S4)
else                                  →  ACCEPT(d) as sub-goal → planner → monitor
```

(* "L1" in the metric mapping refers to the latency tail family L1 in `SPEC_v2.md`,
not architecture layer L1. The architecture/layer names L0–L3 and the latency
metric names L1–L5 are distinct namespaces; see SPEC_v2 §6.)

Properties:
- The default controller (classical planner) is always available; rejection/deferral
  degrades **performance**, never **safety**.
- Every accepted LLM decision is still executed by the planner and verified by the L1
  monitor — the LLM cannot bypass the safety layer.

---

## 6. Relationship to the evaluation framework

The benchmark characterizes the **L3 decision module** to parameterize **L2**:

| Benchmark output | Architectural use |
| --- | --- |
| Unsafe-rate (S1), confidently-unsafe (S2) | Authority ceiling; expected veto-load on the L1 monitor |
| AURC_safety (S4), uncertainty (U1) | Deferral threshold `τ`; proof the trust gate works |
| Deadline-hit (L2-metric), final success rate (R1) | Allowed LLM cadence; fallback frequency |
| Episode success / SPL (P3) with vs without LLM | Value proof: does L3 supervision beat pure A\*, and where |
| Accuracy (P1), step-regret (P2) | L3 decision quality |
| Malformed rate (R4) | L2 reliability gate |

---

## 7. Scope (frozen)

- **In scope:** architecture design; four-family characterization (`SPEC_v2.md`); the
  confidence-gated arbiter; a closed-loop **simulation** (grid rollout) comparing
  pure-A\* vs LLM-supervised episodes.
- **Out of scope (future work):** hardware flight; partial observability / sensor
  fusion; real-time embedded inference; formal verification of the L1 monitor.
