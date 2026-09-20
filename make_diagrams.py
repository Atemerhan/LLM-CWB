#!/usr/bin/env python3
"""Render the two schematic thesis figures (Fig 1 architecture, Fig 2 pipeline).

Unlike `make_figures.py` (data-driven plots from committed results), these are
hand-laid schematics of the frozen design — the L0–L3 / A0–A3 architecture
(`docs/ARCHITECTURE.md`) and the benchmark + experiment pipeline (`run.py`,
`evaluate.py`, `scoring.py`, `u1_selfconsistency.py`, `sim_world.py`). Pure
matplotlib (no data, no model calls); writes PNG + PDF to figures/.

Usage: python make_diagrams.py [--outdir figures]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Palette (consistent with make_figures.py family).
C_LLM = "#ffe0b2"      # untrusted high-level
C_ARB = "#bbdefb"      # arbiter / control
C_SAFE = "#c8e6c9"     # verified / safe-by-construction
C_LOW = "#eeeeee"      # plant / neutral
C_METRIC = "#f8bbd0"   # outputs / metrics
C_HILITE = "#fff59d"   # highlighted (measured ceiling)
EDGE = "#37474f"


def box(ax, cx, cy, w, h, text, fc, fs=8, bold=False, edge=EDGE, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.3, edgecolor=edge, facecolor=fc, linestyle=ls, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p, q, style="-|>", color=EDGE, lw=1.4, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=13, linewidth=lw,
        color=color, linestyle=ls, zorder=1,
        connectionstyle=f"arc3,rad={rad}"))


# --- Fig 1: system architecture -------------------------------------------------

def fig_architecture(outdir):
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.975, "Safety-Bounded Hierarchical LLM Decision Architecture",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(0.32, 0.94, "(L0–L3 stack · Simplex / runtime assurance)",
            ha="center", fontsize=8.5, color="#555")

    xc = 0.32      # left column centre
    w = 0.52
    box(ax, xc, 0.905, 0.30, 0.035, "Mission / semantic goal  (NL, constraints)",
        C_LOW, fs=8)
    box(ax, xc, 0.815, w, 0.085,
        "L3 — LLM  (high-level decision)\n"
        "interpret goal · select sub-goal · when to replan\n"
        "outputs (decision dₜ, uncertainty uₜ)   ~0.1–1 Hz   ·  UNTRUSTED",
        C_LLM, fs=8, bold=False)
    box(ax, xc, 0.685, w, 0.085,
        "L2 — Authority Arbiter / RTA\n"
        "gates:  deadline B · parse-valid · Safe(d|x) · u ≤ τ\n"
        "accept → sub-goal   |   reject / defer / timeout → fallback",
        C_ARB, fs=8)
    box(ax, xc, 0.565, w, 0.05,
        "L2 — Classical Planner  (A*/D*)   ·  optimal given map + goal",
        C_ARB, fs=8)
    box(ax, xc, 0.445, w, 0.075,
        "L1 — Safety Monitor / Verified Executor\n"
        "collision · geofence · kinematic & energy limits — EVERY command\n"
        "HARD SAFETY  (by construction)",
        C_SAFE, fs=8, bold=False)
    box(ax, xc, 0.335, w, 0.045,
        "L0 — Flight Controller  (inner loop)   ·  ~50–1000 Hz", C_LOW, fs=8)
    box(ax, xc, 0.255, w, 0.04, "Vehicle / Environment", C_LOW, fs=8)

    # main downward flow
    arrow(ax, (xc, 0.887), (xc, 0.858))
    arrow(ax, (xc, 0.772), (xc, 0.728))
    arrow(ax, (xc, 0.642), (xc, 0.590))
    arrow(ax, (xc, 0.540), (xc, 0.483))
    arrow(ax, (xc, 0.407), (xc, 0.358))
    arrow(ax, (xc, 0.312), (xc, 0.275))
    ax.text(xc + 0.012, 0.752, "(dₜ, uₜ)", ha="left", fontsize=7, color="#444")

    # fallback loop: arbiter -> planner (right side)
    arrow(ax, (xc + w / 2, 0.685), (xc + w / 2, 0.565), color="#c62828",
          rad=-0.55, lw=1.3)
    ax.text(xc + w / 2 + 0.015, 0.625, "reject /\ndefer →\nfallback", ha="left",
            va="center", fontsize=6.8, color="#c62828")

    # state feedback (far left, up)
    arrow(ax, (xc - w / 2 - 0.015, 0.255), (xc - w / 2 - 0.015, 0.815),
          color="#777", rad=0.0, lw=1.1, ls=(0, (4, 3)))
    ax.text(xc - w / 2 - 0.028, 0.54, "state feedback", rotation=90, va="center",
            ha="center", fontsize=7, color="#777")

    # invariant note
    ax.text(xc, 0.205,
            "Invariant: the LLM never commands an actuator directly.\n"
            "An LLM error ⇒ veto + planner fallback (performance loss), never a collision.",
            ha="center", va="top", fontsize=7.4, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fffde7", ec="#cccccc"))

    # --- authority ladder (right panel) ---
    lx = 0.83
    ax.text(lx, 0.905, "Authority ladder\nA = f(measured trust)", ha="center",
            fontsize=8.5, fontweight="bold")
    rungs = [
        (0.45, "A0  Advisory\nlogged, never acted", C_LOW, False),
        (0.565, "A1  Select-from-safe\npick among planner's SAFE options\n◀ MEASURED CEILING (this work)",
         C_HILITE, True),
        (0.69, "A2  Sub-goal proposal\n(needs a usable trust signal)", C_LOW, False),
        (0.80, "A3  Direct waypoint\n(high trust only)", C_LOW, False),
    ]
    for cy, txt, fc, hl in rungs:
        box(ax, lx, cy, 0.30, 0.078, txt, fc, fs=7.2, bold=hl,
            edge="#f9a825" if hl else EDGE)
    arrow(ax, (lx - 0.18, 0.40), (lx - 0.18, 0.85), color="#555", lw=1.4)
    ax.text(lx - 0.205, 0.625, "increasing authority", rotation=90, va="center",
            ha="center", fontsize=7, color="#555")

    fig.tight_layout()
    _save(fig, outdir, "fig1_architecture")


# --- Fig 2: benchmark + experiment pipeline ------------------------------------

def fig_pipeline(outdir):
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, "Benchmark & Experiment Pipeline",
            ha="center", fontsize=12.5, fontweight="bold")

    # frozen-core dashed enclosure
    ax.add_patch(FancyBboxPatch((0.015, 0.56), 0.97, 0.30,
                 boxstyle="round,pad=0.004", linewidth=1.2, edgecolor="#1565c0",
                 facecolor="#e3f2fd", linestyle=(0, (5, 3)), zorder=0))
    ax.text(0.025, 0.845, "FROZEN core (prompt v0 · dataset · oracle · scoring)",
            ha="left", fontsize=8, color="#1565c0", style="italic")

    y0 = 0.72   # Exp 0 lane
    xs = [0.10, 0.255, 0.41, 0.565, 0.72, 0.885]
    box(ax, xs[0], y0, 0.155, 0.105,
        "Dataset\n1200 samples · seed 42\n8×8 grid · BFS labels\n400/400/400 e/m/h",
        C_LOW, fs=7.5)
    box(ax, xs[1], y0, 0.135, 0.105, "build_prompt\n(prompt v0)\nrender grid + ask\nmove+confidence",
        C_LOW, fs=7.5)
    box(ax, xs[2], y0, 0.135, 0.105, "LLM\nDeepSeek / GLM / Kimi\ntemperature 0",
        C_LLM, fs=7.5)
    box(ax, xs[3], y0, 0.135, 0.105, "parse_response\nextract JSON\nmove + confidence\n(malformed?)",
        C_LOW, fs=7.5)
    box(ax, xs[4], y0, 0.135, 0.105, "BFS oracle\nscore vs ground truth\nsafe / unsafe\ncorrect / regret",
        C_SAFE, fs=7.5)
    box(ax, xs[5], y0, 0.16, 0.105, "Exp 0 metrics\nAcc · ECE · CW\nunsafe-rate (S1/S5)",
        C_METRIC, fs=7.5)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 0.072, y0), (b - 0.070, y0))
    ax.text(0.10, 0.625, "Exp 0 — single-step characterization", ha="left",
            fontsize=8.5, color="#333", fontweight="bold")

    # U1 lane (branch off the LLM node)
    y1 = 0.40
    arrow(ax, (xs[2], 0.668), (xs[2], y1 + 0.055), rad=0.0, color="#6a1b9a")
    u = [
        (0.255, "Resample\nk=10 · T=0.7", C_LLM),
        (0.41, "Majority vote\n→ agreement", C_ARB),
        (0.565, "Risk–coverage\nAURC · AUROC", C_ARB),
        (0.745, "Pre-registered verdict\nFOR / AGAINST  →  AGAINST (3/3)", C_METRIC),
    ]
    for (x, t, c) in u:
        box(ax, x, y1, 0.15, 0.085, t, c, fs=7.5)
    arrow(ax, (0.255 + 0.077, y1), (0.41 - 0.077, y1), color="#6a1b9a")
    arrow(ax, (0.41 + 0.077, y1), (0.565 - 0.077, y1), color="#6a1b9a")
    arrow(ax, (0.565 + 0.077, y1), (0.745 - 0.077, y1), color="#6a1b9a")
    ax.text(0.10, y1, "U1 —\nself-consistency\n(trust signal)", ha="center",
            va="center", fontsize=8, color="#6a1b9a", fontweight="bold")

    # Closed-loop (档 A) lane
    y2 = 0.155
    arrow(ax, (xs[0], 0.668), (xs[0], y2 + 0.05), rad=0.0, color="#2e7d32")
    box(ax, 0.10, y2, 0.15, 0.085, "Frozen grids\nas episodes\n(grid, start, goal)",
        C_LOW, fs=7.5)
    box(ax, 0.355, y2, 0.30, 0.10,
        "Closed-loop rollout  (sim_world)\nL3 LLM → L2 arbiter (A1) → L1 monitor → step\nrepeat to goal / collision / cap",
        C_ARB, fs=7.6)
    box(ax, 0.655, y2, 0.18, 0.085,
        "Mission metrics\nsuccess · SPL\ncollision · veto-load", C_METRIC, fs=7.5)
    box(ax, 0.875, y2, 0.17, 0.085,
        "physics backend\n(AirSim / PhysX)\nsame arbiter+monitor", C_SAFE, fs=7.3,
        ls=(0, (4, 2)))
    arrow(ax, (0.10 + 0.077, y2), (0.355 - 0.152, y2), color="#2e7d32")
    arrow(ax, (0.355 + 0.152, y2), (0.655 - 0.092, y2), color="#2e7d32")
    arrow(ax, (0.655 + 0.092, y2), (0.875 - 0.087, y2), color="#2e7d32", ls=(0, (4, 2)))
    ax.text(0.10, y2 + 0.082, "Track A closed-loop (mission-level)", ha="center",
            fontsize=8, color="#2e7d32", fontweight="bold")

    fig.tight_layout()
    _save(fig, outdir, "fig2_pipeline")


def _save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir}/{stem}.png + .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    fig_architecture(args.outdir)
    fig_pipeline(args.outdir)


if __name__ == "__main__":
    main()
