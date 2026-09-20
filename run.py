"""Experiment 1: Confidently Wrong Detection Benchmark — core engine.

Grid-maze **next-move prediction** with BFS-based ground-truth labeling.

A task is a rectangular grid of open cells (0) and walls (1) with a randomly
placed `goal` and an agent `position`. The question (posed later to an LLM) is:
*from `position`, which single direction (up/down/left/right) moves closer to
the goal?*

Ground truth comes from a BFS shortest-path **distance field** from the goal. A
move is CORRECT iff it strictly decreases the agent's distance to the goal;
moves into walls, off-grid, onto unreachable cells, or onto cells with
equal/greater distance are WRONG. Several moves may be correct, so each label
is the *set* of correct directions.

Each sample also carries difficulty metadata (shortest distance, Manhattan
distance, detour ratio, and an easy/medium/hard bucket) so the evaluation set
can be stratified — see `classify_difficulty`.

No LLM API is used here. Python 3.11 compatible.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass

# 0 = open cell, 1 = wall.
OPEN = 0
WALL = 1

Coord = tuple[int, int]
Grid = list[list[int]]

# Named 4-directional moves -> (row delta, col delta).
MOVES: dict[str, Coord] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

# --- Difficulty thresholds (see SPEC.md). Tuned for ~8x8 grids. -------------
# detour_ratio = shortest_path_distance / manhattan_distance, always >= 1.0.
# ratio == 1.0 means a straight (greedy) path exists; higher means walls force
# a detour, which is exactly where a naive agent fails.
EASY_MAX_DETOUR = 1.0
HARD_MIN_DETOUR = 1.5
EASY_MAX_DISTANCE = 3
HARD_MIN_DISTANCE = 10

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Sample:
    """A single labeled benchmark item (one next-move query)."""

    grid: Grid
    position: Coord
    goal: Coord
    correct_moves: list[str]  # directions that strictly decrease goal-distance
    distance: int             # shortest-path distance position -> goal
    manhattan: int            # |dr| + |dc| (lower bound on distance)
    detour_ratio: float       # distance / manhattan (>= 1.0)
    difficulty: str           # "easy" | "medium" | "hard"


def _in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols


def neighbors(r: int, c: int, rows: int, cols: int) -> list[Coord]:
    """In-bounds 4-directional neighbors of a cell."""
    return [
        (r + dr, c + dc)
        for dr, dc in MOVES.values()
        if _in_bounds(r + dr, c + dc, rows, cols)
    ]


def distance_field(grid: Grid, goal: Coord) -> list[list[int | None]]:
    """BFS shortest-path distance from every cell to `goal`.

    `dist[r][c]` is steps-to-goal, or None for walls/unreachable cells.
    """
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")

    rows, cols = len(grid), len(grid[0])
    gr, gc = goal
    if not _in_bounds(gr, gc, rows, cols):
        raise ValueError("goal out of bounds")

    dist: list[list[int | None]] = [[None] * cols for _ in range(rows)]
    if grid[gr][gc] == WALL:
        return dist

    dist[gr][gc] = 0
    queue: deque[Coord] = deque([goal])
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors(r, c, rows, cols):
            if grid[nr][nc] == OPEN and dist[nr][nc] is None:
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))
    return dist


def cell_distance(grid: Grid, position: Coord, goal: Coord) -> int | None:
    """Shortest-path distance from `position` to `goal`, or None."""
    rows, cols = len(grid), len(grid[0])
    pr, pc = position
    if not _in_bounds(pr, pc, rows, cols):
        raise ValueError("position out of bounds")
    return distance_field(grid, goal)[pr][pc]


def correct_moves(
    grid: Grid,
    position: Coord,
    goal: Coord,
    dist: list[list[int | None]] | None = None,
) -> list[str]:
    """Directions from `position` that strictly decrease distance-to-goal.

    Moves into walls, off-grid, unreachable, or equal/greater-distance cells
    are excluded. Returns directions in canonical DIRECTIONS order.
    """
    rows, cols = len(grid), len(grid[0])
    if dist is None:
        dist = distance_field(grid, goal)

    pr, pc = position
    here = dist[pr][pc]
    if here is None:
        return []

    result: list[str] = []
    for direction in DIRECTIONS:
        dr, dc = MOVES[direction]
        nr, nc = pr + dr, pc + dc
        if not _in_bounds(nr, nc, rows, cols):
            continue
        if grid[nr][nc] == WALL:
            continue
        there = dist[nr][nc]
        if there is not None and there < here:
            result.append(direction)
    return result


def classify_difficulty(distance: int, manhattan: int) -> str:
    """Bucket a sample by shortest-path distance and detour ratio.

    - easy:   no detour needed (ratio <= 1.0) and short (distance <= 3)
    - hard:   large detour (ratio >= 1.5) or long path (distance >= 10)
    - medium: everything in between

    Easy samples are exactly where a greedy "move toward goal" agent succeeds;
    hard samples force navigation around walls.
    """
    if manhattan <= 0:
        raise ValueError("manhattan must be positive (position != goal)")
    ratio = distance / manhattan
    if ratio <= EASY_MAX_DETOUR and distance <= EASY_MAX_DISTANCE:
        return "easy"
    if ratio >= HARD_MIN_DETOUR or distance >= HARD_MIN_DISTANCE:
        return "hard"
    return "medium"


def generate_grid(
    rows: int,
    cols: int,
    wall_prob: float = 0.25,
    rng: random.Random | None = None,
) -> Grid:
    """Random grid; each cell is a wall with probability `wall_prob`.

    Neither goal nor position is fixed here — both are chosen later among the
    open, reachable cells, so no corner is forced open.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if not 0.0 <= wall_prob <= 1.0:
        raise ValueError("wall_prob must be in [0, 1]")

    rng = rng or random.Random()
    return [
        [WALL if rng.random() < wall_prob else OPEN for _ in range(cols)]
        for _ in range(rows)
    ]


def make_sample(
    rows: int,
    cols: int,
    wall_prob: float = 0.25,
    rng: random.Random | None = None,
    difficulty: str | None = None,
    max_attempts: int = 5000,
) -> Sample:
    """Generate one next-move query with a randomly placed goal and position.

    Goal is a random open cell; position is a random reachable, open, non-goal
    cell. If `difficulty` is given, grids are rejection-sampled until the
    sample falls in that bucket. Raises RuntimeError if none is found within
    `max_attempts`.
    """
    if difficulty is not None and difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty: {difficulty}")

    rng = rng or random.Random()
    for _ in range(max_attempts):
        grid = generate_grid(rows, cols, wall_prob, rng)
        open_cells = [
            (r, c) for r in range(rows) for c in range(cols) if grid[r][c] == OPEN
        ]
        if len(open_cells) < 2:
            continue

        goal = rng.choice(open_cells)
        dist = distance_field(grid, goal)
        candidates = [
            (r, c)
            for (r, c) in open_cells
            if (r, c) != goal and dist[r][c] is not None
        ]
        if not candidates:
            continue

        position = rng.choice(candidates)
        pr, pc = position
        gr, gc = goal
        d = dist[pr][pc]
        manhattan = abs(pr - gr) + abs(pc - gc)
        bucket = classify_difficulty(d, manhattan)
        if difficulty is not None and bucket != difficulty:
            continue

        return Sample(
            grid=grid,
            position=position,
            goal=goal,
            correct_moves=correct_moves(grid, position, goal, dist),
            distance=d,
            manhattan=manhattan,
            detour_ratio=round(d / manhattan, 3),
            difficulty=bucket,
        )

    raise RuntimeError(
        f"could not generate difficulty={difficulty!r} sample in "
        f"{max_attempts} attempts (try lower wall_prob or larger grid)"
    )


def _split_counts(n: int, k: int) -> list[int]:
    """Split `n` items into `k` buckets as evenly as possible."""
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def generate_dataset(
    n: int,
    rows: int = 8,
    cols: int = 8,
    wall_prob: float = 0.25,
    seed: int | None = None,
    stratified: bool = True,
) -> list[Sample]:
    """Build `n` labeled samples.

    When `stratified`, samples are balanced across easy/medium/hard buckets so
    accuracy can be reported per difficulty (the validity-critical view).
    """
    rng = random.Random(seed)
    if not stratified:
        return [make_sample(rows, cols, wall_prob, rng) for _ in range(n)]

    counts = _split_counts(n, len(DIFFICULTIES))
    samples: list[Sample] = []
    for difficulty, count in zip(DIFFICULTIES, counts):
        for _ in range(count):
            samples.append(
                make_sample(rows, cols, wall_prob, rng, difficulty=difficulty)
            )
    return samples


# --- JSON persistence -------------------------------------------------------

DATASET_VERSION = 1


def sample_to_dict(sample: Sample) -> dict:
    d = asdict(sample)
    d["position"] = list(sample.position)
    d["goal"] = list(sample.goal)
    return d


def sample_from_dict(d: dict) -> Sample:
    return Sample(
        grid=d["grid"],
        position=tuple(d["position"]),
        goal=tuple(d["goal"]),
        correct_moves=list(d["correct_moves"]),
        distance=d["distance"],
        manhattan=d["manhattan"],
        detour_ratio=d["detour_ratio"],
        difficulty=d["difficulty"],
    )


def save_dataset(path: str, samples: list[Sample], meta: dict | None = None) -> None:
    """Serialize a dataset to JSON so the evaluated set is fixed on disk."""
    payload = {
        "version": DATASET_VERSION,
        "meta": meta or {},
        "samples": [sample_to_dict(s) for s in samples],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_dataset(path: str) -> list[Sample]:
    """Load a dataset from JSON (the canonical evaluation input for Day 2)."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("version") != DATASET_VERSION:
        raise ValueError(
            f"unsupported dataset version: {payload.get('version')!r}"
        )
    return [sample_from_dict(d) for d in payload["samples"]]


def render(sample: Sample) -> str:
    """ASCII rendering: A=agent position, G=goal, #=wall, .=open."""
    chars = {OPEN: ".", WALL: "#"}
    rows = []
    for r, row in enumerate(sample.grid):
        cells = []
        for c, value in enumerate(row):
            if (r, c) == sample.position:
                cells.append("A")
            elif (r, c) == sample.goal:
                cells.append("G")
            else:
                cells.append(chars[value])
        rows.append("".join(cells))
    return "\n".join(rows)


# Default location of the committed evaluation dataset.
DEFAULT_DATASET_PATH = "data/eval_dataset.json"


def main() -> None:
    """Generate and persist the canonical evaluation dataset."""
    rows = cols = 8
    wall_prob = 0.25
    n = 1200
    seed = 42

    samples = generate_dataset(n, rows, cols, wall_prob, seed=seed)
    meta = {
        "rows": rows,
        "cols": cols,
        "wall_prob": wall_prob,
        "seed": seed,
        "n": len(samples),
        "stratified": True,
    }
    import os

    os.makedirs(os.path.dirname(DEFAULT_DATASET_PATH), exist_ok=True)
    save_dataset(DEFAULT_DATASET_PATH, samples, meta)

    counts = {d: 0 for d in DIFFICULTIES}
    for s in samples:
        counts[s.difficulty] += 1
    print(f"wrote {len(samples)} samples -> {DEFAULT_DATASET_PATH}")
    print(f"difficulty counts: {counts}")
    print("\nexample (first sample):")
    s = samples[0]
    print(
        f"position={s.position} goal={s.goal} distance={s.distance} "
        f"manhattan={s.manhattan} detour_ratio={s.detour_ratio} "
        f"difficulty={s.difficulty} correct_moves={s.correct_moves}"
    )
    print(render(s))


if __name__ == "__main__":
    main()
