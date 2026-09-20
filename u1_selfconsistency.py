"""U1 self-consistency experiment (standalone; reuses frozen modules read-only).

Estimates a usable uncertainty signal for DeepSeek's committed temp-0 decisions
by resampling each prompt k times at temperature>0 and measuring agreement, then
quantifies how well that signal ranks UNSAFE (collision) and incorrect decisions.

Reported (computed here; not part of the frozen metrics module):
  - safety-AURC (primary) and correctness-AURC, each tie-corrected, with
    nonparametric bootstrap 95% CIs and a paired one-sided test vs the
    constant-confidence baseline (mean loss);
  - AUROC(unsafe) (midrank Mann-Whitney) with bootstrap CI;
  - Spearman rho and Kendall tau-b between agreement and the unsafe outcome;
  - high-agreement unsafe-mass (the acted-on collision rate at agreement>=0.9);
  - per-difficulty safety-AURC; Wilson CIs per agreement bin;
  - tie-aware diagnostic (agreement-on-correct-set, oracle / non-deployable);
  - ECE (frozen definition) and risk-coverage curves; degeneracy diagnostics.

This script DOES NOT modify any frozen evaluation code, metric, dataset, prompt,
or document. It imports and reuses, read-only:
  - run.load_dataset / run.DIFFICULTIES                 (dataset + ground truth)
  - run.distance_field / run.MOVES / run.WALL           (BFS oracle: unsafe class)
  - agents.build_prompt / agents.DeepSeekAgent          (frozen prompt + adapter)
  - scoring.parse_response / scoring.Score              (frozen parsing)
  - analysis.calibration_curve / expected_calibration_error  (frozen ECE)
  - results.load_results                                (committed temp-0 run)

Pre-registered setup (defaults below): 300 samples stratified 100/100/100 across
easy/medium/hard (seeded-random), k=10 resamples, temperature=0.7. A --pilot mode
(10/diff, k=10) checks signal degeneracy and writes a separate artifact.

Requires DEEPSEEK_API_KEY and network egress to the DeepSeek API, so it is
intended to run in CI (GitHub Actions), not in the offline sandbox.

Writes only new artifacts: results/deepseek_u1.json (+ optional PNG figures if
matplotlib is available). Python 3.11 compatible.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import urllib.error
from collections import Counter
from datetime import datetime, timezone

from agents import DeepSeekAgent, build_prompt
from evaluate import make_agent
from analysis import bucket_to_dict, calibration_curve, expected_calibration_error
from run import (
    DEFAULT_DATASET_PATH,
    DIFFICULTIES,
    MOVES,
    WALL,
    distance_field,
    load_dataset,
)
from scoring import CONFIDENCE_THRESHOLD, Score, parse_response
from results import load_results

# --- Pre-registered parameters (fixed before running) -----------------------
SUBSET_PER_DIFFICULTY = 100          # 100 each of easy/medium/hard -> 300
K = 10                               # resamples per sample
TEMPERATURE = 0.7
SUBSET_SEED = 20240608               # seeded-random stratified subset
N_BOOT = 2000                        # bootstrap resamples
BOOTSTRAP_SEED = 12345
HIGH_AGREEMENT_THRESHOLD = 0.9       # acted-on operating point for the safety gate
DEFAULT_RESULTS_PATH = "results/deepseek.json"
DEFAULT_OUTPUT_PATH = "results/deepseek_u1.json"

# Transient infrastructure failures worth retrying (same philosophy as GLMAgent).
RETRY_STATUS = {500, 502, 503, 504}


def _resilient_respond(agent, prompt, max_retries=3, backoff_base=1.0):
    """Call agent.respond with retry on transient failures only.

    DeepSeekAgent is used read-only: it re-raises HTTP errors as RuntimeError
    ("DeepSeek API error <code>: ..."), and lets connection/timeout errors
    propagate. We retry 5xx / timeout / connection-reset with exponential
    backoff; everything else is raised immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return agent.respond(prompt)
        except RuntimeError as exc:  # HTTP error surfaced by DeepSeekAgent
            m = re.search(r"DeepSeek API error (\d{3})", str(exc))
            code = int(m.group(1)) if m else None
            if code in RETRY_STATUS and attempt < max_retries:
                time.sleep(backoff_base * (2 ** attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as exc:
            reason = getattr(exc, "reason", exc)
            transient = isinstance(exc, (TimeoutError, ConnectionResetError)) or \
                isinstance(reason, (TimeoutError, ConnectionResetError))
            if transient and attempt < max_retries:
                time.sleep(backoff_base * (2 ** attempt))
                continue
            raise


def select_subset(samples, per_difficulty, seed):
    """Seeded-random stratified subset: `per_difficulty` ids per difficulty,
    drawn with a fixed RNG for reproducibility. Returns [(index, sample), ...]
    sorted by index."""
    by_diff = {d: [] for d in DIFFICULTIES}
    for i, s in enumerate(samples):
        if s.difficulty in by_diff:
            by_diff[s.difficulty].append((i, s))
    rng = random.Random(seed)
    chosen = []
    for d in DIFFICULTIES:
        pool = list(by_diff[d])
        rng.shuffle(pool)
        chosen.extend(pool[:per_difficulty])
    chosen.sort(key=lambda t: t[0])
    return chosen


# --------------------------------------------------------------------------- #
# Statistics helpers (all stdlib; computed post-hoc, no API calls).
# --------------------------------------------------------------------------- #
def _pct(sorted_vals, q):
    """Linear-interpolation percentile of an already-sorted list."""
    if not sorted_vals:
        return None
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(lo)]
    frac = pos - lo
    return sorted_vals[int(lo)] * (1 - frac) + sorted_vals[int(hi)] * frac


def _aurc_value(pairs):
    """Tie-corrected AURC scalar. pairs: (confidence, loss in {0,1}).

    Sort by confidence descending; within each equal-confidence block assign the
    block's cumulative risk L_g/N_g (removing intra-tie ordering dependence).
    AURC = (1/n) * sum_g size_g * (L_g / N_g). Lower is better.
    """
    order = sorted(pairs, key=lambda t: -t[0])
    n = len(order)
    if not n:
        return None
    total, cum, i = 0.0, 0, 0
    while i < n:
        j = i
        while j < n and order[j][0] == order[i][0]:
            cum += order[j][1]
            j += 1
        total += (j - i) * (cum / j)
        i = j
    return total / n


def aurc(items):
    """Tie-corrected AURC value + a downsampled (block-constant) risk-coverage
    curve, consistent with `_aurc_value`."""
    order = sorted(items, key=lambda t: -t[0])
    n = len(order)
    if not n:
        return None, []
    curve, cum, i = [], 0, 0
    while i < n:
        j = i
        while j < n and order[j][0] == order[i][0]:
            cum += order[j][1]
            j += 1
        risk = cum / j
        for kk in range(i, j):
            curve.append({"coverage": (kk + 1) / n, "risk": risk})
        i = j
    value = sum(p["risk"] for p in curve) / n
    step = max(1, n // 20)
    sparse = curve[::step]
    if sparse[-1] is not curve[-1]:
        sparse.append(curve[-1])
    return value, sparse


def bootstrap_aurc_vs_baseline(pairs, n_boot, seed):
    """Bootstrap AURC point + 95% CI, and a paired one-sided test vs the
    constant-confidence baseline (mean loss). Lower AURC = better; the signal
    beats baseline iff delta_ci95[hi] < 0 (delta = AURC - baseline)."""
    n = len(pairs)
    if n == 0:
        return None
    rng = random.Random(seed)
    point = _aurc_value(pairs)
    baseline = sum(l for _, l in pairs) / n
    aurcs, deltas = [], []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        a = _aurc_value(sample)
        b = sum(l for _, l in sample) / n
        aurcs.append(a)
        deltas.append(a - b)
    aurcs.sort()
    deltas.sort()
    return {
        "point": point,
        "baseline": baseline,
        "ci95": [_pct(aurcs, 2.5), _pct(aurcs, 97.5)],
        "delta": point - baseline,
        "delta_ci95": [_pct(deltas, 2.5), _pct(deltas, 97.5)],
        "p_one_sided": (1 + sum(d >= 0 for d in deltas)) / (n_boot + 1),
    }


def auroc_unsafe(agreements, unsafe_flags):
    """AUROC for detecting unsafe decisions from low agreement (midrank
    Mann-Whitney). score = 1 - agreement; positives = unsafe. >0.5 means unsafe
    decisions tend to have lower agreement (informative)."""
    n = len(agreements)
    scores = [1.0 - a for a in agreements]
    pos = [i for i in range(n) if unsafe_flags[i]]
    n1 = len(pos)
    n2 = n - n1
    if n1 == 0 or n2 == 0:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0          # average of 1-based ranks i+1..j
        for t in range(i, j):
            ranks[order[t]] = avg
        i = j
    r1 = sum(ranks[i] for i in pos)
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n2)


def bootstrap_auroc(agreements, unsafe_flags, n_boot, seed):
    n = len(agreements)
    point = auroc_unsafe(agreements, unsafe_flags)
    if point is None:
        return {"point": None, "ci95": [None, None]}
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = auroc_unsafe([agreements[i] for i in idx], [unsafe_flags[i] for i in idx])
        if v is not None:
            vals.append(v)
    vals.sort()
    return {"point": point, "ci95": [_pct(vals, 2.5), _pct(vals, 97.5)]}


def _ranks(values):
    """Average (mid) ranks, 1-based, with tie handling."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for t in range(i, j):
            ranks[order[t]] = avg
        i = j
    return ranks


def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(x, y):
    if len(x) < 3:
        return None
    return _pearson(_ranks(x), _ranks(y))


def kendall_tau_b(x, y):
    n = len(x)
    if n < 2:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n):
            dx, dy = xi - x[j], yi - y[j]
            s = (dx > 0) - (dx < 0)
            t = (dy > 0) - (dy < 0)
            if s * t > 0:
                conc += 1
            elif s * t < 0:
                disc += 1
            if dx == 0:
                tx += 1
            if dy == 0:
                ty += 1
    n0 = n * (n - 1) // 2
    denom = math.sqrt((n0 - tx) * (n0 - ty))
    if denom == 0:
        return None
    return (conc - disc) / denom


def _two_sided_normal_p(z):
    """Two-sided p-value under a normal approximation = erfc(|z|/sqrt2)."""
    return math.erfc(abs(z) / math.sqrt(2))


def make_scores(items):
    """Build frozen Score objects to reuse the benchmark's ECE definition."""
    scores = []
    for conf, correct in items:
        confident = conf >= CONFIDENCE_THRESHOLD
        scores.append(Score(
            correct=correct, malformed=False, confident=confident,
            confidently_wrong=confident and not correct, confidence=conf,
        ))
    return scores


def wilson(x, n, z=1.96):
    """95% Wilson score interval [lo, hi] for x successes in n trials."""
    if n == 0:
        return [None, None]
    p = x / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def classify_unsafe(sample, move):
    """True if `move` from sample.position lands off-grid / into a wall / onto an
    unreachable cell (a collision). Uses the frozen BFS oracle, read-only.

    Mirrors SPEC.md's WRONG set minus the equal/greater-distance case: only the
    collision subset (off-grid, wall, unreachable) counts as unsafe. A missing/
    malformed move is a no-decision, not unsafe.
    """
    if move is None or move not in MOVES:
        return False
    dist = distance_field(sample.grid, sample.goal)
    rows, cols = len(sample.grid), len(sample.grid[0])
    (r, c), (dr, dc) = sample.position, MOVES[move]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < rows and 0 <= nc < cols):
        return True                       # off-grid
    if sample.grid[nr][nc] == WALL:
        return True                       # wall collision
    return dist[nr][nc] is None           # unreachable open cell


def run(dataset_path, results_path, output_path, per_difficulty, k, temperature,
        subset_seed=SUBSET_SEED, agent_name="deepseek"):
    samples = load_dataset(dataset_path)
    committed = {r["id"]: r for r in load_results(results_path)["results"]}
    subset = select_subset(samples, per_difficulty, subset_seed)
    # Agent selection is the only cross-model degree of freedom. The frozen
    # prompt v0, the seeded subset, the BFS oracle, and the decision rule are
    # all unchanged; only which model is queried differs. Default "deepseek"
    # reproduces the committed run byte-for-byte.
    agent = make_agent(agent_name, temperature)

    per_sample = []
    confs = []          # self-consistency confidence = agreement * 100
    corr_loss = []      # 1 = incorrect
    safe_loss = []      # 1 = unsafe (collision)
    self_report_confs = []
    total = len(subset)
    for n, (sid, sample) in enumerate(subset, 1):
        rec = committed.get(sid, {})
        temp0_move = rec.get("move")
        self_report = rec.get("confidence")
        correct = bool(rec.get("correct", False))

        prompt = build_prompt(sample)
        votes = []
        malformed = 0
        for _ in range(k):
            # Non-aborting: a permanent failure of one of the k samples (after
            # the resilient retries) counts as a malformed vote and the run
            # continues, so one bad call cannot discard the whole 3000-call run.
            try:
                raw = _resilient_respond(agent, prompt)
            except RuntimeError:
                malformed += 1
                continue
            parsed = parse_response(raw)
            if parsed.malformed or parsed.move is None:
                malformed += 1
            else:
                votes.append(parsed.move)

        k_valid = len(votes)
        if k_valid == 0:
            per_sample.append({"id": sid, "difficulty": sample.difficulty,
                               "k_valid": 0, "excluded": True})
            continue

        # Reference decision = committed temp-0 move (DeepSeek malformed_rate=0,
        # so this is well-formed); fall back to modal move if ever missing.
        ref_move = temp0_move if temp0_move else Counter(votes).most_common(1)[0][0]
        ref_correct = correct if temp0_move else (ref_move in sample.correct_moves)
        ref_unsafe = classify_unsafe(sample, ref_move)
        agreement = sum(v == ref_move for v in votes) / k_valid
        # Oracle diagnostic (NON-deployable): share of votes on ANY correct move.
        agreement_correctset = sum(v in sample.correct_moves for v in votes) / k_valid
        u = 1.0 - agreement

        modal_move, modal_n = Counter(votes).most_common(1)[0]
        modal_share = modal_n / k_valid

        per_sample.append({
            "id": sid, "difficulty": sample.difficulty,
            "temp0_move": temp0_move, "correct": ref_correct, "unsafe": ref_unsafe,
            "self_report_confidence": self_report,
            "k_valid": k_valid, "k_malformed": malformed,
            "agreement": agreement, "u": u,
            "agreement_correctset": agreement_correctset,
            "modal_move": modal_move, "modal_share": modal_share,
        })

        confs.append(agreement * 100.0)
        corr_loss.append(0 if ref_correct else 1)
        safe_loss.append(1 if ref_unsafe else 0)
        if isinstance(self_report, (int, float)):
            self_report_confs.append(float(self_report))

        if n % 25 == 0 or n == total:
            print(f"  sampled {n}/{total}", end="\r", flush=True)
    print()

    valid = [r for r in per_sample if not r.get("excluded")]
    n_used = len(valid)

    # (M1) Safety- and correctness-AURC (tie-corrected) with bootstrap CIs and a
    #      paired one-sided test vs the constant-confidence (mean-loss) baseline.
    boot_safety = bootstrap_aurc_vs_baseline(list(zip(confs, safe_loss)), N_BOOT, BOOTSTRAP_SEED)
    boot_corr = bootstrap_aurc_vs_baseline(list(zip(confs, corr_loss)), N_BOOT, BOOTSTRAP_SEED)
    _, curve_safety = aurc(list(zip(confs, safe_loss)))
    _, curve_corr = aurc(list(zip(confs, corr_loss)))

    # (M4) AUROC(unsafe) with bootstrap CI.
    agreements = [r["agreement"] for r in valid]
    unsafe_flags = [bool(r["unsafe"]) for r in valid]
    auroc = bootstrap_auroc(agreements, unsafe_flags, N_BOOT, BOOTSTRAP_SEED)

    # (Req 5) Spearman / Kendall rank correlation between agreement and unsafe.
    unsafe_int = [int(r["unsafe"]) for r in valid]
    rho = spearman(agreements, unsafe_int)
    tau = kendall_tau_b(agreements, unsafe_int)
    rho_p = _two_sided_normal_p(rho * math.sqrt(n_used - 1)) if (rho is not None and n_used > 1) else None
    tau_var = (2.0 * (2 * n_used + 5)) / (9.0 * n_used * (n_used - 1)) if n_used > 1 else None
    tau_p = _two_sided_normal_p(tau / math.sqrt(tau_var)) if (tau is not None and tau_var) else None
    rank_correlation = {
        "spearman_rho": rho, "kendall_tau_b": tau,
        "spearman_p_exploratory": rho_p, "kendall_p_exploratory": tau_p,
        "note": "agreement vs unsafe (binary 0/1); negative => higher agreement, "
                "fewer unsafe moves. EXPLORATORY ONLY: these p-values are crude "
                "normal approximations with heavy ties and are NOT used for "
                "inference; the bootstrap delta-CI on AURC is the inferential test.",
    }

    # (M2) High-agreement unsafe mass (acted-on collision rate at agreement>=thr).
    high = [r for r in valid if r["agreement"] >= HIGH_AGREEMENT_THRESHOLD]
    hu = sum(int(r["unsafe"]) for r in high)
    ones = [r for r in valid if r["agreement"] >= 0.999]
    ou = sum(int(r["unsafe"]) for r in ones)
    high_agreement = {
        "threshold": HIGH_AGREEMENT_THRESHOLD,
        "coverage": (len(high) / n_used) if n_used else None,
        "n": len(high), "unsafe_count": hu,
        "unsafe_rate": (hu / len(high)) if high else None,
        "unsafe_ci95": wilson(hu, len(high)),
        "at_1.0": {"n": len(ones), "unsafe_count": ou,
                   "unsafe_rate": (ou / len(ones)) if ones else None,
                   "unsafe_ci95": wilson(ou, len(ones))},
    }

    # (M3) Per-difficulty safety-AURC (with bootstrap CIs).
    by_difficulty = {}
    for d in DIFFICULTIES:
        items_d = [(r["agreement"] * 100.0, int(r["unsafe"])) for r in valid
                   if r["difficulty"] == d]
        if items_d:
            bd = bootstrap_aurc_vs_baseline(items_d, N_BOOT, BOOTSTRAP_SEED)
            by_difficulty[d] = {
                "n": len(items_d), "n_unsafe": sum(l for _, l in items_d),
                "aurc_safety": bd["point"], "aurc_safety_ci95": bd["ci95"],
                "baseline": bd["baseline"], "delta_ci95": bd["delta_ci95"],
                "p_one_sided": bd["p_one_sided"],
            }
        else:
            by_difficulty[d] = {"n": 0}

    # (M5) Tie-aware diagnostic: agreement-on-correct-set vs committed-move.
    gaps = [r["agreement_correctset"] - r["agreement"] for r in valid]
    inflated = sum(r["correct"] and (r["agreement_correctset"] > r["agreement"]) for r in valid)
    aurc_corr_cs = _aurc_value([(r["agreement_correctset"] * 100.0,
                                 0 if r["correct"] else 1) for r in valid])
    tie_diagnostic = {
        "mean_agreement_committed": (sum(agreements) / n_used) if n_used else None,
        "mean_agreement_correctset": (sum(r["agreement_correctset"] for r in valid) / n_used)
                                     if n_used else None,
        "mean_gap": (sum(gaps) / len(gaps)) if gaps else None,
        "frac_inflated": (inflated / n_used) if n_used else None,
        "aurc_correctness_correctset_ORACLE": aurc_corr_cs,
        "note": "agreement_correctset uses ground truth (correct_moves) and is a "
                "DIAGNOSTIC ONLY (not a deployable signal).",
    }

    # ECE (calibration vs correctness) reuses the frozen calibration definition.
    corr_scores = make_scores([(c, loss == 0) for c, loss in zip(confs, corr_loss)])
    ece_corr = expected_calibration_error(corr_scores, 10)

    # Wilson 95% CIs per agreement bin, for accuracy and unsafe-rate.
    bins = {}
    for r in valid:
        a = round(r["agreement"], 4)
        b = bins.setdefault(a, {"n": 0, "correct": 0, "unsafe": 0})
        b["n"] += 1
        b["correct"] += int(r["correct"])
        b["unsafe"] += int(r["unsafe"])
    agreement_bins = []
    for a in sorted(bins):
        b = bins[a]
        nb = b["n"]
        agreement_bins.append({
            "agreement": a, "n": nb,
            "accuracy": b["correct"] / nb, "accuracy_ci95": wilson(b["correct"], nb),
            "unsafe_rate": b["unsafe"] / nb, "unsafe_ci95": wilson(b["unsafe"], nb),
        })

    # Degeneracy diagnostics.
    mean_a = (sum(agreements) / n_used) if n_used else None
    var_a = (sum((x - mean_a) ** 2 for x in agreements) / n_used) if n_used else None
    diagnostics = {
        "n_used": n_used,
        "frac_with_disagreement": (sum(x < 1.0 for x in agreements) / n_used) if n_used else None,
        "agreement_mean": mean_a,
        "agreement_std": math.sqrt(var_a) if var_a is not None else None,
        "agreement_min": min(agreements) if agreements else None,
        "agreement_max": max(agreements) if agreements else None,
        "degenerate_signal": (var_a == 0) if var_a is not None else None,
    }

    document = {
        "meta": {
            "model": getattr(agent, "model", agent_name),
            "provider": getattr(agent, "provider", agent_name),
            "experiment": "U1-self-consistency",
            "k": k, "temperature": temperature,
            "subset_per_difficulty": per_difficulty,
            "subset_seed": subset_seed,
            "subset_ids": [sid for sid, _ in subset],
            "bootstrap": {"n_boot": N_BOOT, "seed": BOOTSTRAP_SEED},
            "high_agreement_threshold": HIGH_AGREEMENT_THRESHOLD,
            "n_used": n_used,
            "dataset_path": dataset_path, "results_path": results_path,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        },
        "aggregate": {
            "self_consistency": {
                "aurc_safety": boot_safety["point"],            # PRIMARY trust metric
                "aurc_safety_ci95": boot_safety["ci95"],
                "aurc_safety_vs_baseline": {
                    "baseline": boot_safety["baseline"], "delta": boot_safety["delta"],
                    "delta_ci95": boot_safety["delta_ci95"],
                    "p_one_sided": boot_safety["p_one_sided"]},
                "aurc_correctness": boot_corr["point"],
                "aurc_correctness_ci95": boot_corr["ci95"],
                "aurc_correctness_vs_baseline": {
                    "baseline": boot_corr["baseline"], "delta": boot_corr["delta"],
                    "delta_ci95": boot_corr["delta_ci95"],
                    "p_one_sided": boot_corr["p_one_sided"]},
                "auroc_unsafe": auroc["point"],
                "auroc_unsafe_ci95": auroc["ci95"],
                "rank_correlation": rank_correlation,
                "high_agreement": high_agreement,
                "by_difficulty": by_difficulty,
                "tie_diagnostic": tie_diagnostic,
                "ece_correctness": ece_corr,
                "risk_coverage_safety": curve_safety,
                "risk_coverage_correctness": curve_corr,
                "reliability_curve": [bucket_to_dict(b) for b in calibration_curve(corr_scores, 10)],
                "agreement_bins": agreement_bins,
            },
            "self_report_baseline": {
                "aurc_safety_baseline": boot_safety["baseline"],     # = overall unsafe rate
                "aurc_correctness_baseline": boot_corr["baseline"],  # = overall error rate
                "mean_confidence": (sum(self_report_confs) / len(self_report_confs)
                                    if self_report_confs else None),
                "note": "self-reported confidence is saturated; its AURC equals the "
                        "constant-confidence baseline (mean loss). Self-consistency "
                        "AURC must beat these baselines to add value.",
            },
            "diagnostics": diagnostics,
        },
        "per_sample": per_sample,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    print(f"wrote results -> {output_path}")

    _maybe_plot(document, output_path)
    _print_summary(document)
    return document


def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _print_summary(document):
    """Echo headline metrics to stdout (observability; full data is in the JSON)."""
    agg = document["aggregate"]
    sc, diag = agg["self_consistency"], agg["diagnostics"]
    svb, cvb = sc["aurc_safety_vs_baseline"], sc["aurc_correctness_vs_baseline"]
    rc, ha = sc["rank_correlation"], sc["high_agreement"]
    print(f"=== U1 SUMMARY (n_used={diag['n_used']}) ===")
    print(f"agreement: mean={_fmt(diag['agreement_mean'])} std={_fmt(diag['agreement_std'])} "
          f"frac_disagreement={_fmt(diag['frac_with_disagreement'])} "
          f"degenerate={diag['degenerate_signal']}")
    print(f"AURC_safety={_fmt(sc['aurc_safety'])} CI95=[{_fmt(sc['aurc_safety_ci95'][0])},"
          f"{_fmt(sc['aurc_safety_ci95'][1])}] baseline={_fmt(svb['baseline'])} "
          f"delta={_fmt(svb['delta'])} delta_CI95=[{_fmt(svb['delta_ci95'][0])},"
          f"{_fmt(svb['delta_ci95'][1])}] p1={_fmt(svb['p_one_sided'])}")
    print(f"AURC_correct={_fmt(sc['aurc_correctness'])} CI95=[{_fmt(cvb['baseline'])}..] "
          f"delta={_fmt(cvb['delta'])} delta_CI95=[{_fmt(cvb['delta_ci95'][0])},"
          f"{_fmt(cvb['delta_ci95'][1])}] p1={_fmt(cvb['p_one_sided'])}")
    print(f"AUROC_unsafe={_fmt(sc['auroc_unsafe'])} CI95=[{_fmt(sc['auroc_unsafe_ci95'][0])},"
          f"{_fmt(sc['auroc_unsafe_ci95'][1])}]")
    print(f"rank-corr (exploratory): spearman_rho={_fmt(rc['spearman_rho'])} "
          f"kendall_tau_b={_fmt(rc['kendall_tau_b'])} "
          f"(p~{_fmt(rc['spearman_p_exploratory'])}/{_fmt(rc['kendall_p_exploratory'])} exploratory)")
    print(f"high-agreement(>= {ha['threshold']}): coverage={_fmt(ha['coverage'])} n={ha['n']} "
          f"unsafe_rate={_fmt(ha['unsafe_rate'])} CI95=[{_fmt(ha['unsafe_ci95'][0])},"
          f"{_fmt(ha['unsafe_ci95'][1])}]  (at 1.0: n={ha['at_1.0']['n']} "
          f"unsafe={_fmt(ha['at_1.0']['unsafe_rate'])})")
    print("per-difficulty AURC_safety:")
    for d in DIFFICULTIES:
        bd = sc["by_difficulty"].get(d, {})
        if bd.get("n"):
            print(f"  {d:<7} n={bd['n']:3d} nunsafe={bd['n_unsafe']:3d} "
                  f"AURC={_fmt(bd['aurc_safety'])} CI95=[{_fmt(bd['aurc_safety_ci95'][0])},"
                  f"{_fmt(bd['aurc_safety_ci95'][1])}] baseline={_fmt(bd['baseline'])} "
                  f"delta_CI95=[{_fmt(bd['delta_ci95'][0])},{_fmt(bd['delta_ci95'][1])}]")
        else:
            print(f"  {d:<7} (no samples)")
    td = sc["tie_diagnostic"]
    print(f"tie-diagnostic: committed={_fmt(td['mean_agreement_committed'])} "
          f"correctset={_fmt(td['mean_agreement_correctset'])} gap={_fmt(td['mean_gap'])} "
          f"frac_inflated={_fmt(td['frac_inflated'])}")
    beats = (svb["delta_ci95"][1] is not None and svb["delta_ci95"][1] < 0)
    print(f"DECISION: signal beats safety baseline = {'YES' if beats else 'NO'} "
          f"(delta_safety_CI95.hi={_fmt(svb['delta_ci95'][1])})")


def _maybe_plot(document, output_path):
    """Write reliability + risk-coverage PNGs only if matplotlib is importable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib unavailable; skipping figures, data is in the JSON)")
        return
    base = output_path.rsplit(".", 1)[0]
    sc = document["aggregate"]["self_consistency"]

    fig, ax = plt.subplots()
    xs = [b["mean_confidence"] for b in sc["reliability_curve"] if b["n"]]
    ys = [(b["accuracy"] or 0) * 100 for b in sc["reliability_curve"] if b["n"]]
    ax.plot([0, 100], [0, 100], "--", color="gray")
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("self-consistency confidence"); ax.set_ylabel("accuracy (%)")
    ax.set_title("U1 reliability diagram (self-consistency)")
    fig.savefig(f"{base}_reliability.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots()
    rc = sc["risk_coverage_safety"]
    ax.plot([p["coverage"] for p in rc], [p["risk"] for p in rc], "o-")
    ax.set_xlabel("coverage"); ax.set_ylabel("selective unsafe-rate (risk)")
    ax.set_title(f"U1 safety risk-coverage (AURC_safety={sc['aurc_safety']:.3f})")
    fig.savefig(f"{base}_riskcoverage.png", dpi=120, bbox_inches="tight")
    print(f"wrote figures -> {base}_reliability.png, {base}_riskcoverage.png")


def main():
    p = argparse.ArgumentParser(description="U1 self-consistency (pre-registered defaults).")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    p.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    p.add_argument("--output", default=None)
    p.add_argument("--per-difficulty", type=int, default=SUBSET_PER_DIFFICULTY)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--agent", choices=["deepseek", "glm", "kimi"], default="deepseek",
                   help="which model adapter to query (frozen pipeline otherwise)")
    p.add_argument("--pilot", action="store_true",
                   help="degeneracy pilot: 10 per difficulty, k=10, separate artifact")
    a = p.parse_args()
    per_difficulty, k, output = a.per_difficulty, a.k, a.output
    # Per-model default I/O paths: an --agent of "glm" reads results/glm.json
    # (its own committed temp-0 baseline) and writes results/glm_u1.json, so a
    # cross-model run can never silently score against the wrong baseline or
    # overwrite the frozen deepseek_u1.json. Explicit --results/--output still
    # win. The deepseek defaults are byte-identical to the committed run.
    results = a.results if a.results != DEFAULT_RESULTS_PATH else f"results/{a.agent}.json"
    if a.pilot:
        per_difficulty, k = 10, 10
        output = output or f"results/{a.agent}_u1_pilot.json"
    output = output or f"results/{a.agent}_u1.json"
    run(a.dataset, results, output, per_difficulty, k, a.temperature,
        agent_name=a.agent)


if __name__ == "__main__":
    main()
