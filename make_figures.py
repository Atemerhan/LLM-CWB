#!/usr/bin/env python3
"""Render thesis figures from the committed frozen result artifacts.

Read-only: loads results/{deepseek,glm,kimi}.json (Exp 0) and *_u1.json (U1) and
writes PNG figures to figures/. Touches no benchmark/scoring/prompt/dataset/U1
code — pure visualization of already-committed evidence. Requires matplotlib.

Usage: python make_figures.py [--outdir figures]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [
    ("deepseek", "DeepSeek (deepseek-chat)", "#1f77b4"),
    ("glm", "GLM (glm-4-flash)", "#ff7f0e"),
    ("kimi", "Kimi (moonshot-v1-8k)", "#2ca02c"),
]


def load(name, suffix=""):
    with open(f"results/{name}{suffix}.json") as f:
        return json.load(f)


def risk_coverage(u1):
    """Risk (cumulative unsafe-rate) vs coverage, ordering by agreement desc."""
    ps = u1.get("per_sample", [])
    pairs = [
        (p["agreement"], 1 if p.get("unsafe") else 0)
        for p in ps
        if p.get("agreement") is not None and p.get("unsafe") is not None
    ]
    pairs.sort(key=lambda t: t[0], reverse=True)
    n = len(pairs)
    cov, risk, cum = [], [], 0
    for k, (_, u) in enumerate(pairs, 1):
        cum += u
        cov.append(k / n)
        risk.append(cum / k)
    return cov, risk


def fig_calibration(outdir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, label, c in MODELS:
        s = load(name)["summary"]["overall"]
        conf, acc = s["mean_confidence"] / 100.0, s["accuracy"]
        ax.scatter([conf], [acc], s=120, color=c, label=f"{label}: conf {s['mean_confidence']:.0f}, acc {acc:.3f}", zorder=3)
        ax.annotate(f"gap={conf-acc:.2f}", (conf, acc), textcoords="offset points", xytext=(-10, 10), fontsize=8)
    ax.set_xlabel("Mean reported confidence"); ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_title("Fig 3 — Calibration: confidence vs accuracy\n(all models saturate confidence at 1.0)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig3_calibration.png", dpi=150); plt.close(fig)


def fig_ece(outdir):
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [m[1].split(" (")[0] for m in MODELS]
    eces = [load(m[0])["summary"]["overall"]["ece"] for m in MODELS]
    accs = [load(m[0])["summary"]["overall"]["accuracy"] for m in MODELS]
    x = range(len(MODELS))
    ax.bar([i - 0.2 for i in x], eces, width=0.4, label="ECE", color="#d62728")
    ax.bar([i + 0.2 for i in x], [1 - a for a in accs], width=0.4, label="error rate (1-acc)", color="#7f7f7f")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("value"); ax.set_ylim(0, 1)
    ax.set_title("Fig 5 — ECE equals error rate\n(confidence saturated → single bucket)")
    for i, e in enumerate(eces): ax.text(i - 0.2, e + 0.01, f"{e:.3f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig5_ece_comparison.png", dpi=150); plt.close(fig)


def fig_confidence_hist(outdir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, label, c in MODELS:
        confs = [r["confidence"] for r in load(name)["results"]
                 if not r.get("malformed") and r.get("confidence") is not None]
        ax.hist(confs, bins=[i for i in range(0, 105, 5)], alpha=0.5, label=label, color=c)
    ax.set_xlabel("Reported confidence"); ax.set_ylabel("count")
    ax.set_title("Fig 6 — Confidence distribution\n(degenerate spike at 100 for all models)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig6_confidence_hist.png", dpi=150); plt.close(fig)


def fig_risk_coverage(outdir):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, label, c in MODELS:
        u1 = load(name, "_u1")
        cov, risk = risk_coverage(u1)
        base = u1["aggregate"]["self_consistency"]["aurc_safety_vs_baseline"]["baseline"]
        au = u1["aggregate"]["self_consistency"]["aurc_safety"]
        ax.plot(cov, risk, color=c, label=f"{label}: AURC={au:.3f}")
        ax.axhline(base, color=c, ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Coverage (fraction acted on, ordered by agreement)")
    ax.set_ylabel("Risk (unsafe-rate among acted-on)")
    ax.set_title("Fig 7 — Safety risk–coverage (U1)\nflat ≈ no usable safety signal; dotted = base unsafe-rate")
    ax.legend(fontsize=8); ax.set_ylim(0, 0.6)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig7_risk_coverage.png", dpi=150); plt.close(fig)


def fig_auroc_gate(outdir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    labels = [m[1].split(" (")[0] for m in MODELS]
    x = range(len(MODELS))
    aurocs, los, his = [], [], []
    for name, _, _ in MODELS:
        sc = load(name, "_u1")["aggregate"]["self_consistency"]
        aurocs.append(sc["auroc_unsafe"]); los.append(sc["auroc_unsafe_ci95"][0]); his.append(sc["auroc_unsafe_ci95"][1])
    err = [[a - l for a, l in zip(aurocs, los)], [h - a for a, h in zip(aurocs, his)]]
    a1.bar(list(x), aurocs, yerr=err, capsize=5, color=[m[2] for m in MODELS])
    a1.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    a1.set_xticks(list(x)); a1.set_xticklabels(labels); a1.set_ylim(0.4, 0.75)
    a1.set_ylabel("AUROC(unsafe)"); a1.set_title("(a) Discrimination vs chance"); a1.legend(fontsize=8)
    for i, v in enumerate(aurocs): a1.text(i, his[i] + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    # gate: unsafe@>=0.9 vs baseline
    gate, base = [], []
    for name, _, _ in MODELS:
        ha = load(name, "_u1")["aggregate"]["self_consistency"]["high_agreement"]
        gate.append(ha["unsafe_rate"]); base.append(load(name, "_u1")["aggregate"]["self_consistency"]["aurc_safety_vs_baseline"]["baseline"])
    a2.bar([i - 0.2 for i in x], gate, width=0.4, label="unsafe@agree≥0.9", color="#d62728")
    a2.bar([i + 0.2 for i in x], base, width=0.4, label="baseline unsafe-rate", color="#7f7f7f")
    a2.set_xticks(list(x)); a2.set_xticklabels(labels); a2.set_ylabel("unsafe-rate")
    a2.set_title("(b) High-agreement not safer"); a2.legend(fontsize=8)
    fig.suptitle("Fig 8 — Self-consistency as a safety gate (U1): fails on all three models")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig8_auroc_gate.png", dpi=150); plt.close(fig)


def fig_cross_model(outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [m[1].split(" (")[0] for m in MODELS]
    acc = [load(m[0])["summary"]["overall"]["accuracy"] for m in MODELS]
    ece = [load(m[0])["summary"]["overall"]["ece"] for m in MODELS]
    cw = [load(m[0])["summary"]["overall"]["confidently_wrong_rate"] for m in MODELS]
    auroc = [load(m[0], "_u1")["aggregate"]["self_consistency"]["auroc_unsafe"] for m in MODELS]
    x = range(len(MODELS)); w = 0.2
    ax.bar([i - 1.5 * w for i in x], acc, w, label="accuracy", color="#1f77b4")
    ax.bar([i - 0.5 * w for i in x], ece, w, label="ECE", color="#d62728")
    ax.bar([i + 0.5 * w for i in x], cw, w, label="confidently-wrong rate", color="#9467bd")
    ax.bar([i + 1.5 * w for i in x], auroc, w, label="AUROC(unsafe)", color="#2ca02c")
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
    ax.set_title("Fig 9 — Cross-model comparison (n=3)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig9_cross_model.png", dpi=150); plt.close(fig)


def fig_agreement_hist(outdir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, label, c in MODELS:
        ag = [p["agreement"] for p in load(name, "_u1")["per_sample"] if p.get("agreement") is not None]
        ax.hist(ag, bins=[i / 10 for i in range(0, 11)], alpha=0.5, label=label, color=c)
    ax.set_xlabel("Self-consistency agreement"); ax.set_ylabel("count")
    ax.set_title("Fig 6b — Agreement distribution (U1)\nKimi notably less self-consistent")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig6b_agreement_hist.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fig_calibration(a.outdir)
    fig_ece(a.outdir)
    fig_confidence_hist(a.outdir)
    fig_risk_coverage(a.outdir)
    fig_auroc_gate(a.outdir)
    fig_cross_model(a.outdir)
    fig_agreement_hist(a.outdir)
    print(f"wrote figures to {a.outdir}/")


if __name__ == "__main__":
    main()
