"""
Deterministic 2048 game mechanics and feature extraction.
"""

from __future__ import annotations

import math
import random

ACTIONS_2048 = [0, 1, 2, 3]
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


def _rng(seed: int | None = None) -> random.Random:
    return random.Random(seed)


def clone_board(board: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in board]


def new_board_2048(seed: int | None = None, rng: random.Random | None = None) -> list[list[int]]:
    source = rng if rng is not None else _rng(seed)
    board = [[0] * 4 for _ in range(4)]
    add_tile(board, rng=source)
    add_tile(board, rng=source)
    return board


def add_tile(board: list[list[int]], rng: random.Random | None = None) -> None:
    source = rng if rng is not None else random
    empties = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if not empties:
        return
    row, col = source.choice(empties)
    board[row][col] = 4 if source.random() < 0.1 else 2


def slide_row_left(row: list[int]) -> tuple[list[int], int]:
    tiles = [value for value in row if value != 0]
    merged: list[int] = []
    score = 0
    idx = 0

    while idx < len(tiles):
        if idx + 1 < len(tiles) and tiles[idx] == tiles[idx + 1]:
            value = tiles[idx] * 2
            merged.append(value)
            score += value
            idx += 2
        else:
            merged.append(tiles[idx])
            idx += 1

    return merged + [0] * (4 - len(merged)), score


def move_2048(board: list[list[int]], action: int) -> tuple[list[list[int]], int, bool]:
    """Return ``(new_board, score_gain, moved)`` without mutating input."""
    moved_board = clone_board(board)
    total_score = 0
    moved = False

    if action == 2:
        for row in range(4):
            new_row, score = slide_row_left(moved_board[row])
            moved = moved or new_row != moved_board[row]
            moved_board[row] = new_row
            total_score += score
    elif action == 3:
        for row in range(4):
            reversed_row, score = slide_row_left(moved_board[row][::-1])
            new_row = reversed_row[::-1]
            moved = moved or new_row != moved_board[row]
            moved_board[row] = new_row
            total_score += score
    elif action == 0:
        for col in range(4):
            current_col = [moved_board[row][col] for row in range(4)]
            new_col, score = slide_row_left(current_col)
            moved = moved or new_col != current_col
            for row in range(4):
                moved_board[row][col] = new_col[row]
            total_score += score
    elif action == 1:
        for col in range(4):
            current_col = [moved_board[row][col] for row in range(4)][::-1]
            shifted_col, score = slide_row_left(current_col)
            new_col = shifted_col[::-1]
            moved = moved or new_col != [moved_board[row][col] for row in range(4)]
            for row in range(4):
                moved_board[row][col] = new_col[row]
            total_score += score

    return moved_board, total_score, moved


def valid_actions_2048(board: list[list[int]]) -> list[int]:
    return [action for action in ACTIONS_2048 if move_2048(board, action)[2]]


def is_game_over_2048(board: list[list[int]]) -> bool:
    return not valid_actions_2048(board)


def flatten(board: list[list[int]]) -> list[int]:
    return [board[row][col] for row in range(4) for col in range(4)]


def max_tile(board: list[list[int]]) -> int:
    return max(flatten(board))


def empty_count(board: list[list[int]]) -> int:
    return flatten(board).count(0)


def _log_tile(value: int) -> int:
    return int(math.log2(value)) if value > 0 else 0


def monotonicity_score(board: list[list[int]]) -> int:
    monotonic = 0
    for row in range(4):
        values = board[row]
        if all(values[idx] >= values[idx + 1] for idx in range(3)) or all(
            values[idx] <= values[idx + 1] for idx in range(3)
        ):
            monotonic += 1
    for col in range(4):
        values = [board[row][col] for row in range(4)]
        if all(values[idx] >= values[idx + 1] for idx in range(3)) or all(
            values[idx] <= values[idx + 1] for idx in range(3)
        ):
            monotonic += 1
    return monotonic


def smoothness_score(board: list[list[int]]) -> int:
    penalty = 0
    for row in range(4):
        for col in range(4):
            value = board[row][col]
            if value == 0:
                continue
            current = _log_tile(value)
            if col < 3 and board[row][col + 1] != 0:
                penalty += abs(current - _log_tile(board[row][col + 1]))
            if row < 3 and board[row + 1][col] != 0:
                penalty += abs(current - _log_tile(board[row + 1][col]))
    return max(0, 24 - penalty)


def snake_score(board: list[list[int]]) -> int:
    weights = [
        15, 14, 13, 12,
        8, 9, 10, 11,
        7, 6, 5, 4,
        0, 1, 2, 3,
    ]
    return sum(value * weights[idx] for idx, value in enumerate(flatten(board)))


def heuristic_value(board: list[list[int]]) -> float:
    tile = max_tile(board)
    return (
        empty_count(board) * 25.0
        + monotonicity_score(board) * 20.0
        + smoothness_score(board) * 3.0
        + (_log_tile(tile) * 45.0 if tile > 0 else 0.0)
        + math.log2(snake_score(board) + 1) * 8.0
    )


def board_features(board: list[list[int]]) -> tuple[int, int, int, int, int]:
    """
    Feature key for tabular Q-learning:
    (max log2 tile, empty bucket, monotonic rows/cols, smoothness bucket, snake bucket).
    """
    tile = max_tile(board)
    snake = snake_score(board)
    smooth = smoothness_score(board)
    return (
        _log_tile(tile),
        min(empty_count(board), 8),
        min(monotonicity_score(board), 8),
        min(smooth // 3, 8),
        min(int(math.log2(snake + 1)) // 2, 10) if snake > 0 else 0,
    )
