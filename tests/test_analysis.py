from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from analysis import run_comparison
from game2048.game2048_rl import Game2048QLearner
from sudoku.sudoku_rl import SudokuQLearner


def test_run_comparison_returns_all_project_methods() -> None:
    sudoku_agent = SudokuQLearner("easy", seed=11)
    sudoku_agent.train(episodes=2)
    game_agent = Game2048QLearner(seed=12)
    game_agent.train(episodes=2)

    comparison = run_comparison(
        sudoku_agent,
        game_agent,
        difficulty="easy",
        seeds=[101],
        game_max_moves=20,
    )

    sudoku_methods = {row["method"] for row in comparison["sudoku"]["summary"]}
    game_methods = {row["method"] for row in comparison["game2048"]["summary"]}

    assert sudoku_methods == {"Pure Backtracking", "MRV Heuristic", "Random MRV", "RL Q + MRV"}
    assert game_methods == {"RL Q Policy", "Expectimax", "Hybrid Q + Expectimax", "Random Policy"}
