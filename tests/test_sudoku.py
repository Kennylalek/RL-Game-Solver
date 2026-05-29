from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from sudoku.sudoku_baselines import solve_backtracking, solve_mrv, solve_random_mrv
from sudoku.sudoku_rl import SudokuQLearner
from sudoku.sudoku_utils import count_solutions, generate_sudoku, mrv_cell, state_features


def test_generate_sudoku_returns_unique_solution() -> None:
    puzzle, solution = generate_sudoku(35, seed=42, ensure_unique=True)

    assert len(puzzle) == 81
    assert puzzle.count(0) == 35
    assert solution.count(0) == 0
    assert count_solutions(puzzle, limit=2) == 1


def test_sudoku_baselines_solve_known_generated_puzzle() -> None:
    puzzle, _ = generate_sudoku(35, seed=7, ensure_unique=True)

    assert solve_backtracking(puzzle).solved is True
    assert solve_mrv(puzzle).solved is True
    assert solve_random_mrv(puzzle, seed=7).solved is True


def test_sudoku_q_update_uses_max_next_action_value() -> None:
    puzzle, _ = generate_sudoku(35, seed=13, ensure_unique=True)
    agent = SudokuQLearner("easy", seed=1)
    state = state_features(puzzle)
    action_idx, action_candidates = mrv_cell(puzzle)
    action = (action_idx, action_candidates[0])
    next_board = puzzle[:]
    next_board[action[0]] = action[1]
    next_state = state_features(next_board)
    next_idx, next_candidates = mrv_cell(next_board)
    best_next = next_candidates[0]

    agent.q[(next_state, next_idx, best_next)] = 2.0
    agent.update(state, action, reward=1.0, next_board=next_board, done=False)

    assert agent.q[(state, action[0], action[1])] == 0.87


def test_sudoku_training_and_solving_emit_live_progress() -> None:
    puzzle, _ = generate_sudoku(35, seed=21, ensure_unique=True)
    agent = SudokuQLearner("easy", seed=2)
    training_rows = []
    live_steps = []

    agent.train(episodes=2, progress_callback=training_rows.append)
    steps, solved = agent.solve_with_steps(puzzle, progress_callback=live_steps.append)

    assert len(training_rows) == 2
    assert len(live_steps) == len(steps)
    assert live_steps[0]["message"] == "Puzzle loaded"
    assert solved is True
