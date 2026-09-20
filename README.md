# LLM-CWB — Detecting Confidently Wrong Reasoning in LLMs via a Grid Navigation Benchmark

> **一句话**：用 1200 个栅格导航决策样本实测发现，DeepSeek / GLM / Kimi 在答错时**平均自报置信度仍是 100%**——
> 自报置信度完全饱和，不含任何可用的自我可信信号。因此本项目不依赖模型自评，而是外置一个分级权限安全层（A0–A3）：
> 在栅格运动学、真实 LLM 闭环、AirSim 物理引擎三种保真度下，A1 门控把碰撞率**稳定压到 0**。

**TL;DR** — LLMs asked to navigate a grid are wrong 42–69% of the time *while reporting 100% confidence*.
Self-reported confidence is therefore useless as a safety gate. This repo ships (a) a reproducible
benchmark that measures exactly that, and (b) an external authority-gating layer that makes the
closed-loop system safe regardless of how badly the model reasons.

---

## 1. The finding: confidence is saturated, not calibrated

Benchmark: 1200 grid-navigation decisions, ground truth from BFS shortest path, three difficulty tiers.
Each model returns a move *and* a self-reported confidence.

| Model | n | Accuracy | Mean self-reported confidence | ECE | Confidently-wrong rate |
|---|---:|---:|---:|---:|---:|
| `deepseek-chat` | 1200 | 0.575 | **100.0** | 0.425 | 0.425 |
| `moonshot-v1-8k` (Kimi) | 1200 | 0.335 | **100.0** | 0.665 | 0.665 |
| `glm-4-flash` | 1200 | 0.315 | **100.0** | 0.685 | 0.685 |

The decisive column is not accuracy — it is that `right_low` and `wrong_low` are **both exactly 0** for all
three models. No model ever reported low confidence, on any sample. ECE collapses to the error rate,
because confidence is a constant.

**Consequence for system design:** a self-confidence threshold cannot gate anything. Safety has to come
from outside the model. That is what the rest of this repo is.

## 2. The safety layer: authority levels A0–A3

Instead of trusting the model's self-assessment, every LLM output passes a runtime monitor that adjudicates
it against the world model before execution:

| Level | Model authority |
|---|---|
| A0 | advisory only — classical planner executes |
| **A1** | model proposes, monitor vetoes any unsafe command (**the level this work validates**) |
| A2 | model executes with post-hoc audit |
| A3 | unrestricted |

The benchmark result above is the argument for capping authority at **A1**: without a usable self-signal,
no higher level is defensible.

## 3. Closed-loop results across three fidelity levels

| Fidelity | n | Policy / model | Ungated | Under A1 |
|---|---:|---|---|---|
| Kinematic grid | 1200 | random | succ 0.06 · **coll 0.94** | succ **0.79** · **coll 0** |
| Kinematic grid | 1200 | greedy | succ 0.41 · coll 0.59 | succ 0.65 · **coll 0** |
| Real LLM | 100 | `deepseek-chat` | succ 0.37 · coll 0.31 | succ 0.57 · **coll 0** |
| Real LLM | 100 | `glm-4-flash` | succ 0.04 · coll 0.96 | succ 0.38 · **coll 0** |
| Real LLM | 100 | `moonshot-v1-8k` | succ 0.05 · coll 0.93 | succ 0.34 · **coll 0** |
| AirSim / PhysX | 8 | greedy / random | coll 0.125 / 0.875 | **coll 0 / 0** |

Collision rate under A1 is **0 in every row, at every fidelity level** — the safety property is structural,
not statistical. Task success is not: it tracks how good the underlying policy is, which is the honest
result (a safety layer cannot make a bad planner good, only harmless).

AirSim runs are `n=8` — episode count was reduced after RPC instability under heavy ungated-random
collisions. It is a validity check that the architecture survives a real physics engine, not a
statistical claim.

Raw numbers: [`results/`](results/) · Figure: [`figures/fig10_closed_loop.png`](figures/fig10_closed_loop.png)

## 4. Reproduce

Requires Python 3.11. Core benchmark needs **no third-party packages** (stdlib only); `matplotlib`
is needed to render figures. Heavy optional deps (`airsim`, `torch`, `transformers`, `scikit-learn`)
are lazily imported so CI and the core path stay light.

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=...        # or GLM_API_KEY / KIMI_API_KEY

python evaluate.py --provider deepseek         # benchmark one model -> results/<model>.json
python run_embodied_llm.py --provider deepseek # closed-loop, ungated vs A1 -> results/embodied_llm_*.json
python make_fig10.py                           # render the cross-fidelity figure
python -m unittest discover -p 'test_*.py'     # 12 test modules
```

No API key is stored in this repo — every provider reads its key from an environment variable.

## 5. Figures

| | |
|---|---|
| `fig1_architecture.png` | four-layer system: perception → LLM decision → classical planner → control, with the external safety layer |
| `fig5_ece_comparison.png` | ECE across models |
| `fig6_confidence_hist.png` | the saturation result — every mass at confidence = 100 |
| `fig8_auroc_gate.png` | AUROC of self-confidence as a gating signal |
| `fig10_closed_loop.png` | collision rate, ungated vs A1, across all three fidelity levels |

## 6. Layout

```
agents.py           provider clients (DeepSeek / GLM / Kimi), prompt construction
run.py              grid world + BFS ground truth, dataset generation
sim_world.py        kinematic simulator, safety adjudication, AirSim backend (lazy import)
embodied_sim.py     closed-loop episode runner, authority arms
scoring.py          response parsing, confidence extraction
evaluate.py         benchmark driver  -> results/<model>.json
analysis.py         accuracy / ECE / confidently-wrong categorisation
u1_selfconsistency.py  self-consistency probe
u2_probe.py         local small-model probes (phi-3.5-mini, qwen2.5-3b)
test_*.py           12 unit-test modules
.github/workflows/  17 CI workflows — per-model benchmarks, smoke tests, figure rendering
results/            committed raw results — every number in this README comes from here
figures/            rendered figures
docs/               design notes and per-experiment reports
```

## 7. Status

Undergraduate thesis pre-study (2026.01–2026.06), continuing as the formal thesis project.
Author: Yerhan Awuzibieke (叶尔汗·阿吾孜别克), Shandong University of Technology.
