# UNSAFE_BREAKDOWN.md — S5 unsafe sub-types (Exp 0, n=1200)

Read-only re-analysis of committed `results/<model>.json` via the frozen
BFS oracle. Unsafe = off-grid ∨ wall ∨ unreachable (collision proxy).
Rates are over well-formed moves.

| Model | safe | off-grid | wall | unreachable | unsafe-rate |
| --- | --- | --- | --- | --- | --- |
| DeepSeek deepseek-chat | 882 | 48 | 270 | 0 | **0.265** (318/1200) |
| GLM glm-4-flash | 778 | 163 | 259 | 0 | **0.352** (422/1200) |
| Kimi moonshot-v1-8k | 801 | 132 | 267 | 0 | **0.333** (399/1200) |

## Reading
- **wall** collisions and **off-grid** steps are *true collisions* (the L1 monitor must veto these); **unreachable** = stepping onto an open but cut-off cell (a softer failure).
- This is the SPEC_v2 §4.3 **S5** metric, computed read-only from the frozen Exp 0 artifacts; it refines the safety story for Ch4/Ch5 and sharpens the per-step collision proxy behind S1/S3.
