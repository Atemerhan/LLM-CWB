"""Closed-loop LLM decision arm for the 档 A embodied simulation.

Unlike the single-step benchmark, a closed-loop episode visits states that are
NOT in the frozen dataset (the agent moves), so decisions must be obtained
**live** from the model — they cannot be replayed from `results/*.json`. This
runner drives full episodes through the L2 arbiter / L1 monitor with a real
LLM as the high-level (L3) policy, reusing the frozen `build_prompt` (prompt v0)
and `parse_response` (identical to Exp 0 / U1), and writes
`results/embodied_llm_<model>.json`.

It is non-aborting (a failed/malformed call = no decision → the arbiter handles
it: under A1 it is unsafe → planner fallback; under UNGATED it stalls the
episode) and memoizes per-(grid,state) decisions so revisited cells and the
shared cache across arms keep the call count (and cost) down.

Arms: `planner` (pure A*, no calls), `a1` (LLM bounded to safe options + L1),
and optionally `ungated` (LLM executed verbatim — the danger ablation, safe to
run only in simulation). The headline to confirm with a real LLM is that **A1
holds the collision rate at 0** while `ungated` collides at the per-step unsafe
rate compounded over the trajectory.

Run locally/CI:
  python run_embodied_llm.py --agent deepseek --limit 100 --arms a1,planner,ungated
No model calls happen at import; the agent is built only in `main`.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

import embodied_sim as E
import run
import sim_world as W
from agents import build_prompt
from evaluate import make_agent
from scoring import parse_response


def sample_at(grid: run.Grid, pos: tuple[int, int],
              goal: tuple[int, int]) -> run.Sample:
    """Build a Sample for an arbitrary in-loop state so prompt v0 renders it."""
    dist = run.distance_field(grid, goal)
    d = dist[pos[0]][pos[1]]
    d = d if d is not None else 0
    manhattan = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    bucket = run.classify_difficulty(d, manhattan) if manhattan > 0 else "easy"
    return run.Sample(
        grid=grid, position=pos, goal=goal,
        correct_moves=run.correct_moves(grid, pos, goal, dist),
        distance=d, manhattan=manhattan,
        detour_ratio=round(d / max(manhattan, 1), 3), difficulty=bucket,
    )


class LLMPolicy:
    """Grid policy `fn(grid,pos,goal)->move` backed by a live LLM agent.

    Memoizes decisions by (grid, pos, goal); counts calls and malformed/no
    -decision events. Never raises — a failed call yields no decision (None),
    which the arbiter treats as unsafe (A1 → fallback) or a stall (UNGATED).
    """

    def __init__(self, agent) -> None:
        self.agent = agent
        self.cache: dict[tuple, Optional[str]] = {}
        self._gridkey: dict[int, tuple] = {}
        self.calls = 0
        self.malformed = 0

    def _key(self, grid, pos, goal):
        gk = self._gridkey.get(id(grid))
        if gk is None:
            gk = tuple(tuple(row) for row in grid)
            self._gridkey[id(grid)] = gk
        return (gk, pos, goal)

    def __call__(self, grid, pos, goal) -> Optional[str]:
        key = self._key(grid, pos, goal)
        if key in self.cache:
            return self.cache[key]
        prompt = build_prompt(sample_at(grid, pos, goal))
        try:
            raw = self.agent.respond(prompt)
        except Exception as exc:                       # non-aborting
            raw = ""
            print(f"  [warn] decision call failed: {exc}", flush=True)
        self.calls += 1
        parsed = parse_response(raw)
        move = None if parsed.malformed else parsed.move
        if move is None:
            self.malformed += 1
        self.cache[key] = move
        return move


ARM_AUTHORITY = {
    "planner": E.Authority.PLANNER_ONLY,
    "a1": E.Authority.A1_SELECT_FROM_SAFE,
    "ungated": E.Authority.UNGATED,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop LLM decision arm.")
    p.add_argument("--agent", choices=["greedy", "deepseek", "glm", "kimi"],
                   default="deepseek")
    p.add_argument("--limit", type=int, default=100,
                   help="number of frozen episodes to roll out")
    p.add_argument("--arms", default="a1,planner,ungated",
                   help="comma-separated subset of: planner,a1,ungated")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_AUTHORITY:
            raise SystemExit(f"unknown arm: {a} (choose from {list(ARM_AUTHORITY)})")

    worlds = W.grid_worlds_from_dataset(limit=args.limit)
    agent = make_agent(args.agent, args.temperature)
    model = getattr(agent, "model", agent.name)
    provider = getattr(agent, "provider", agent.name)
    print(f"closed-loop LLM arm: agent={args.agent} model={model} "
          f"episodes={len(worlds)} arms={arms}\n")

    # One shared policy across arms so identical states are queried once.
    llm = LLMPolicy(agent)
    llm_world_policy = W.grid_policy(llm)
    out_arms = []
    for arm in arms:
        if arm == "planner":
            policy = W.grid_policy(E.oracle_policy)     # no LLM calls
        else:
            policy = llm_world_policy
        results = W.run_arm(worlds, policy, ARM_AUTHORITY[arm])
        s = E.summarize(results)
        s["arm"] = arm
        out_arms.append(s)
        print(f"{arm:<8} succ={s['success_rate']:.3f} spl={s['spl']:.3f} "
              f"coll={s['collision_rate']:.3f} veto={s['veto_rate']:.3f} "
              f"(llm_calls={llm.calls} malformed={llm.malformed})")

    payload = {
        "experiment": "embodied_archA_llm_closed_loop",
        "model": model, "provider": provider, "agent": args.agent,
        "temperature": args.temperature, "episodes": len(worlds),
        "prompt_version": "v0",
        "llm_calls": llm.calls, "malformed_decisions": llm.malformed,
        "note": "Closed-loop; decisions live (cannot replay frozen single-step "
                "results). Frozen dataset reused as episodes; v0 baseline untouched.",
        "arms": out_arms,
    }
    out = args.output or f"results/embodied_llm_{args.agent}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
