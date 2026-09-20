#!/usr/bin/env python3
"""Fig 10 — closed-loop collision rate: ungated vs A1, across three fidelity
levels (kinematic n=1200 · real-LLM n=100 · AirSim/PhysX n=8). Data-driven from
the committed results/embodied_*.json. The figure's message: A1 holds collision
at 0 everywhere, while the ungated controller collides — the constructive
safety-by-construction result. Requires matplotlib.

Usage: python make_fig10.py [--outdir figures]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return json.load(f)


def arm(arms, **kv):
    for a in arms:
        if all(a.get(k) == v for k, v in kv.items()):
            return a
    return None


def collect():
    """(label, ungated_collision, a1_collision, group) rows, in display order."""
    rows = []
    # Kinematic reference (results/embodied_baseline.json)
    kb = load("results/embodied_baseline.json")["arms"]
    for pol in ("greedy", "random"):
        u = arm(kb, authority="ungated", policy=pol)
        a = arm(kb, authority="a1", policy=pol)
        rows.append((f"{pol}", u["collision_rate"], a["collision_rate"], "Kinematic\nn=1200"))
    # Real-LLM (results/embodied_llm_*.json)
    llm_files = [("results/embodied_llm_deepseek.json", "DeepSeek"),
                 ("results/embodied_llm_glm.json", "GLM-4-flash"),
                 ("results/embodied_llm_kimi.json", "Kimi"),
                 ("results/embodied_llm_glm_glm-5.1_.json", "GLM-5.1")]
    for path, name in llm_files:
        if not os.path.exists(path):
            continue
        a = load(path)["arms"]
        u = arm(a, arm="ungated")
        a1 = arm(a, arm="a1")
        rows.append((name, u["collision_rate"], a1["collision_rate"], "Real LLM\nn=100"))
    # AirSim (results/embodied_airsim_arms.json)
    if os.path.exists("results/embodied_airsim_arms.json"):
        ar = load("results/embodied_airsim_arms.json")["arms"]
        for pol in ("greedy", "random"):
            u = arm(ar, arm=f"ungated_{pol}")
            a1 = arm(ar, arm=f"a1_{pol}")
            rows.append((f"{pol}", u["collision_rate"], a1["collision_rate"],
                         "AirSim PhysX\nn=8"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    rows = collect()

    labels = [r[0] for r in rows]
    ung = [r[1] for r in rows]
    a1 = [r[2] for r in rows]
    groups = [r[3] for r in rows]

    x = list(range(len(rows)))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    b1 = ax.bar([i - w / 2 for i in x], ung, w, label="ungated",
                color="#e53935", edgecolor="#7f1d1d")
    b2 = ax.bar([i + w / 2 for i in x], a1, w, label="A1 (select-from-safe)",
                color="#2e7d32", edgecolor="#1b4332")
    ax.set_ylabel("Collision rate")
    ax.set_ylim(0, 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax.set_title("Fig 10  Closed-loop collision rate: ungated vs A1 across three fidelity levels",
                 fontsize=11)
    ax.legend(loc="center", bbox_to_anchor=(0.685, 0.60), framealpha=0.96, fontsize=9)
    ax.axhline(0, color="#333", lw=0.8)

    # value labels
    for rect, v in list(zip(b1, ung)) + list(zip(b2, a1)):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7,
                color="#7f1d1d" if rect in b1 else "#1b4332")

    # group separators + labels
    boundaries, prev = [], None
    for i, g in enumerate(groups):
        if g != prev:
            boundaries.append(i)
            prev = g
    for b in boundaries[1:]:
        ax.axvline(b - 0.5, color="#bbb", ls="--", lw=1, ymax=0.86)
    for gi, b in enumerate(boundaries):
        end = boundaries[gi + 1] if gi + 1 < len(boundaries) else len(rows)
        ax.text((b + end - 1) / 2, 1.12, groups[b], ha="center", va="center",
                fontsize=9, color="#1565c0", fontweight="bold")

    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"fig10_closed_loop.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.outdir}/fig10_closed_loop.png + .pdf")
    for r in rows:
        print(f"  {r[3]:<18} {r[0]:<12} ungated={r[1]:.3f}  a1={r[2]:.3f}")


if __name__ == "__main__":
    main()
