#!/usr/bin/env python3
"""
Run reproducible project experiments and write report-ready artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from analysis import run_comparison, write_experiment_artifacts
from game2048.game2048_rl import Game2048QLearner
from sudoku.sudoku_rl import SudokuQLearner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL Game Solver comparison experiments.")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--sudoku-episodes", type=int, default=200)
    parser.add_argument("--game-episodes", type=int, default=500)
    parser.add_argument("--game-max-moves", type=int, default=1000)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "experiments"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]

    sudoku_agent = SudokuQLearner(difficulty=args.difficulty, seed=29)
    sudoku_history = sudoku_agent.train(episodes=args.sudoku_episodes)
    sudoku_agent.save()

    game_agent = Game2048QLearner(seed=31)
    game_history = game_agent.train(episodes=args.game_episodes)
    game_agent.save()

    comparison = run_comparison(
        sudoku_agent,
        game_agent,
        difficulty=args.difficulty,
        seeds=seeds,
        game_max_moves=args.game_max_moves,
    )
    artifacts = write_experiment_artifacts(
        comparison,
        args.output_dir,
        sudoku_history=sudoku_history,
        game_history=game_history,
    )

    print("Experiment complete")
    for key, value in artifacts.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
