"""
Experiment and comparison utilities for the project report and UI dashboard.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

from game2048.game2048_rl import Game2048QLearner
from game2048.game2048_utils import max_tile
from sudoku.sudoku_baselines import solve_backtracking, solve_mrv, solve_random_mrv
from sudoku.sudoku_rl import SudokuQLearner
from sudoku.sudoku_utils import empty_count_for_difficulty, generate_sudoku


@dataclass(frozen=True)
class ComparisonConfig:
    difficulty: str = "medium"
    seeds: tuple[int, ...] = (101, 202, 303)
    sudoku_unique: bool = True
    sudoku_max_nodes: int = 100_000
    game_max_moves: int = 1000


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 3) if values else 0.0


def _success_rate(values: list[bool]) -> float:
    return round(sum(1 for value in values if value) / len(values), 3) if values else 0.0


def evaluate_sudoku_methods(
    agent: SudokuQLearner | None,
    config: ComparisonConfig,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    num_empty = empty_count_for_difficulty(config.difficulty)

    for seed in config.seeds:
        puzzle, _ = generate_sudoku(
            num_empty,
            seed=seed,
            ensure_unique=config.sudoku_unique,
        )

        baseline_results = [
            solve_backtracking(puzzle, max_nodes=config.sudoku_max_nodes),
            solve_mrv(puzzle, max_nodes=config.sudoku_max_nodes),
            solve_random_mrv(puzzle, seed=seed, max_nodes=config.sudoku_max_nodes),
        ]
        for result in baseline_results:
            records.append({
                "domain": "Sudoku",
                "seed": seed,
                "method": result.method,
                "solved": result.solved,
                "steps": result.steps,
                "duration_ms": result.duration_ms,
            })

        if agent is not None and agent.trained:
            started = time.perf_counter()
            steps, solved = agent.solve_with_steps(puzzle)
            records.append({
                "domain": "Sudoku",
                "seed": seed,
                "method": "RL Q + MRV",
                "solved": solved,
                "steps": len(steps),
                "duration_ms": (time.perf_counter() - started) * 1000,
            })

    summaries: list[dict[str, Any]] = []
    for method in sorted({record["method"] for record in records}):
        method_records = [record for record in records if record["method"] == method]
        summaries.append({
            "method": method,
            "runs": len(method_records),
            "success_rate": _success_rate([bool(record["solved"]) for record in method_records]),
            "avg_steps": _mean([float(record["steps"]) for record in method_records]),
            "avg_duration_ms": _mean([float(record["duration_ms"]) for record in method_records]),
        })

    summaries.sort(key=lambda row: (-float(row["success_rate"]), float(row["avg_steps"])))
    return {
        "difficulty": config.difficulty,
        "empty_cells": num_empty,
        "runs": len(config.seeds),
        "records": records,
        "summary": summaries,
    }


def evaluate_2048_methods(
    agent: Game2048QLearner,
    config: ComparisonConfig,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    policies = [
        ("RL Q Policy", "q"),
        ("Expectimax", "expectimax"),
        ("Hybrid Q + Expectimax", "hybrid"),
        ("Random Policy", "random"),
    ]

    for seed in config.seeds:
        for method, policy in policies:
            started = time.perf_counter()
            steps = agent.play_episode(policy=policy, seed=seed, max_moves=config.game_max_moves)
            duration_ms = (time.perf_counter() - started) * 1000
            final = steps[-1]
            records.append({
                "domain": "2048",
                "seed": seed,
                "method": method,
                "policy": policy,
                "score": int(final["score"]),
                "moves": len(steps) - 1,
                "max_tile": max_tile(final["board"]),
                "duration_ms": duration_ms,
            })

    summaries: list[dict[str, Any]] = []
    for method in sorted({record["method"] for record in records}):
        method_records = [record for record in records if record["method"] == method]
        summaries.append({
            "method": method,
            "runs": len(method_records),
            "avg_score": _mean([float(record["score"]) for record in method_records]),
            "best_score": max(int(record["score"]) for record in method_records),
            "avg_max_tile": _mean([float(record["max_tile"]) for record in method_records]),
            "best_max_tile": max(int(record["max_tile"]) for record in method_records),
            "avg_moves": _mean([float(record["moves"]) for record in method_records]),
            "avg_duration_ms": _mean([float(record["duration_ms"]) for record in method_records]),
        })

    summaries.sort(key=lambda row: (-float(row["avg_score"]), -float(row["avg_max_tile"])))
    return {
        "runs": len(config.seeds),
        "records": records,
        "summary": summaries,
    }


def run_comparison(
    sudoku_agent: SudokuQLearner | None,
    game_agent: Game2048QLearner,
    *,
    difficulty: str = "medium",
    seeds: list[int] | tuple[int, ...] | None = None,
    game_max_moves: int = 1000,
) -> dict[str, Any]:
    seed_tuple = tuple(seeds or [101, 202, 303])
    config = ComparisonConfig(
        difficulty=difficulty,
        seeds=seed_tuple,
        game_max_moves=game_max_moves,
    )
    return {
        "config": {
            "difficulty": config.difficulty,
            "seeds": list(config.seeds),
            "game_max_moves": config.game_max_moves,
        },
        "sudoku": evaluate_sudoku_methods(sudoku_agent, config),
        "game2048": evaluate_2048_methods(game_agent, config),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_training_plots(
    output_dir: Path,
    *,
    sudoku_history: list[dict[str, Any]],
    game_history: list[dict[str, Any]],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    try:
        cache_root = Path(__file__).resolve().parent / "artifacts" / ".plot-cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    if sudoku_history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            [row["episode"] for row in sudoku_history],
            [row["reward"] for row in sudoku_history],
            label="Reward",
        )
        ax.plot(
            [row["episode"] for row in sudoku_history],
            [row["filled"] for row in sudoku_history],
            label="Filled Cells",
        )
        ax.set_title("Sudoku Q-learning training curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward / filled cells")
        ax.legend()
        fig.tight_layout()
        target = output_dir / "sudoku_training_curve.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        written.append(str(target))

    if game_history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            [row["episode"] for row in game_history],
            [row["score"] for row in game_history],
            label="Score",
        )
        ax.plot(
            [row["episode"] for row in game_history],
            [row["max_tile"] for row in game_history],
            label="Max Tile",
        )
        ax.set_title("2048 Q-learning training curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score / max tile")
        ax.legend()
        fig.tight_layout()
        target = output_dir / "game2048_training_curve.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        written.append(str(target))

    return written


def write_experiment_artifacts(
    comparison: dict[str, Any],
    output_dir: str | Path,
    *,
    sudoku_history: list[dict[str, Any]],
    game_history: list[dict[str, Any]],
) -> dict[str, Any]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    sudoku_csv = target_dir / "sudoku_comparison.csv"
    game_csv = target_dir / "game2048_comparison.csv"
    summary_json = target_dir / "summary.json"

    write_csv(sudoku_csv, comparison["sudoku"]["records"])
    write_csv(game_csv, comparison["game2048"]["records"])
    summary_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    plots = write_training_plots(
        target_dir,
        sudoku_history=sudoku_history,
        game_history=game_history,
    )

    return {
        "sudoku_csv": str(sudoku_csv),
        "game2048_csv": str(game_csv),
        "summary_json": str(summary_json),
        "plots": plots,
    }
