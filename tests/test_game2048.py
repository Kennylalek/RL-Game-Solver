from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from game2048.game2048_rl import Game2048QLearner
from game2048.game2048_utils import board_features, move_2048, slide_row_left


def test_slide_row_left_merges_once_per_pair() -> None:
    row, score = slide_row_left([2, 2, 4, 4])

    assert row == [4, 8, 0, 0]
    assert score == 12


def test_move_2048_left_is_pure_and_scores_exactly() -> None:
    board = [
        [2, 0, 2, 4],
        [0, 4, 4, 0],
        [2, 2, 2, 2],
        [0, 0, 0, 0],
    ]

    new_board, score_gain, moved = move_2048(board, 2)

    assert moved is True
    assert score_gain == 20
    assert new_board == [
        [4, 4, 0, 0],
        [8, 0, 0, 0],
        [4, 4, 0, 0],
        [0, 0, 0, 0],
    ]
    assert board[0] == [2, 0, 2, 4]


def test_2048_agent_supports_all_evaluation_policies() -> None:
    agent = Game2048QLearner(seed=3)
    agent.train(episodes=3)

    for policy in ("q", "expectimax", "hybrid", "random"):
        steps = agent.play_episode(policy=policy, seed=5, max_moves=20)
        assert len(steps) > 1
        assert steps[0]["policy"] == policy
        assert "explain" in steps[0]


def test_2048_training_and_play_emit_live_progress() -> None:
    agent = Game2048QLearner(seed=9)
    training_rows = []
    live_steps = []

    agent.train(episodes=2, progress_callback=training_rows.append)
    steps = agent.play_episode(policy="q", seed=10, max_moves=10, progress_callback=live_steps.append)

    assert len(training_rows) == 2
    assert len(live_steps) == len(steps)
    assert live_steps[0]["message"] == "Game started"


def test_board_features_are_stable() -> None:
    board = [
        [16, 8, 4, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    assert board_features(board) == (4, 8, 8, 7, 4)
