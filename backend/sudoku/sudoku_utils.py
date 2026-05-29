"""
Deterministic Sudoku utilities.

The public generator returns puzzles with a unique solution by default. Training
can opt out of uniqueness checks when a large number of random boards is needed.
"""

from __future__ import annotations

import random
from typing import Iterable

BOARD_SIZE = 81
GRID_SIZE = 9
BOX_SIZE = 3

DIFFICULTY_EMPTY_COUNTS = {
    "easy": 35,
    "medium": 45,
    "hard": 55,
}


def empty_count_for_difficulty(difficulty: str) -> int:
    return DIFFICULTY_EMPTY_COUNTS.get(difficulty, DIFFICULTY_EMPTY_COUNTS["medium"])


def sudoku_empty_count(board: list[int], diff: str) -> int:
    return empty_count_for_difficulty(diff)


def _rng(seed: int | None = None) -> random.Random:
    return random.Random(seed)


def _row_values(board: list[int], row: int) -> Iterable[int]:
    start = row * GRID_SIZE
    return board[start:start + GRID_SIZE]


def _col_values(board: list[int], col: int) -> Iterable[int]:
    return (board[col + GRID_SIZE * row] for row in range(GRID_SIZE))


def _box_values(board: list[int], row: int, col: int) -> Iterable[int]:
    box_row = (row // BOX_SIZE) * BOX_SIZE
    box_col = (col // BOX_SIZE) * BOX_SIZE
    for dr in range(BOX_SIZE):
        for dc in range(BOX_SIZE):
            yield board[(box_row + dr) * GRID_SIZE + box_col + dc]


def is_valid_placement(board: list[int], idx: int, val: int) -> bool:
    if not 1 <= val <= GRID_SIZE:
        return False

    row, col = divmod(idx, GRID_SIZE)
    previous = board[idx]
    board[idx] = 0
    valid = (
        val not in _row_values(board, row)
        and val not in _col_values(board, col)
        and val not in _box_values(board, row, col)
    )
    board[idx] = previous
    return valid


def is_valid_board(board: list[int]) -> bool:
    if len(board) != BOARD_SIZE:
        return False
    for idx, val in enumerate(board):
        if val == 0:
            continue
        if not is_valid_placement(board, idx, val):
            return False
    return True


def get_candidates(board: list[int], idx: int) -> list[int]:
    """Return valid digits for an empty cell."""
    if board[idx] != 0:
        return []

    row, col = divmod(idx, GRID_SIZE)
    used = set(_row_values(board, row))
    used.update(_col_values(board, col))
    used.update(_box_values(board, row, col))
    used.discard(0)
    return [val for val in range(1, GRID_SIZE + 1) if val not in used]


def mrv_cell(board: list[int]) -> tuple[int, list[int]]:
    """Return the empty cell with minimum remaining values."""
    best_idx = -1
    best_candidates: list[int] = []
    best_count = GRID_SIZE + 1

    for idx in range(BOARD_SIZE):
        if board[idx] != 0:
            continue

        candidates = get_candidates(board, idx)
        if not candidates:
            return idx, []
        if len(candidates) < best_count:
            best_idx = idx
            best_candidates = candidates
            best_count = len(candidates)
            if best_count == 1:
                break

    return best_idx, best_candidates


def first_empty_cell(board: list[int]) -> tuple[int, list[int]]:
    for idx, val in enumerate(board):
        if val == 0:
            return idx, get_candidates(board, idx)
    return -1, []


def state_features(board: list[int]) -> tuple[int, int, int, int]:
    """
    Compact feature vector for tabular Q-learning:
    (empty bucket, naked singles, double candidates, contradictions).
    """
    empty = board.count(0)
    singles = 0
    doubles = 0
    conflicts = 0

    for idx in range(BOARD_SIZE):
        if board[idx] != 0:
            continue
        count = len(get_candidates(board, idx))
        if count == 0:
            conflicts += 1
        elif count == 1:
            singles += 1
        elif count == 2:
            doubles += 1

    return (empty // 5, min(singles, 9), min(doubles, 9), min(conflicts, 3))


def count_solutions(board: list[int], limit: int = 2) -> int:
    """Count solutions up to a small limit; enough to prove uniqueness."""
    working = board[:]
    if not is_valid_board(working):
        return 0

    found = 0

    def search() -> None:
        nonlocal found
        if found >= limit:
            return

        idx, candidates = mrv_cell(working)
        if idx == -1:
            found += 1
            return
        if not candidates:
            return

        for val in candidates:
            working[idx] = val
            search()
            working[idx] = 0
            if found >= limit:
                return

    search()
    return found


def solve_sudoku_board(
    puzzle: list[int],
    *,
    use_mrv: bool = True,
    randomize: bool = False,
    seed: int | None = None,
    max_nodes: int = 100_000,
) -> tuple[list[int], bool, int]:
    """Solve a Sudoku puzzle with deterministic or randomized backtracking."""
    board = puzzle[:]
    rng = _rng(seed)
    nodes = 0

    def select_cell() -> tuple[int, list[int]]:
        return mrv_cell(board) if use_mrv else first_empty_cell(board)

    def search() -> bool:
        nonlocal nodes
        if nodes >= max_nodes:
            return False

        idx, candidates = select_cell()
        if idx == -1:
            return True
        if not candidates:
            return False

        if randomize:
            rng.shuffle(candidates)

        for val in candidates:
            nodes += 1
            board[idx] = val
            if search():
                return True
            board[idx] = 0

        return False

    solved = is_valid_board(board) and search()
    return board, solved, nodes


def _fill_complete_board(board: list[int], rng: random.Random, pos: int = 0) -> bool:
    if pos == BOARD_SIZE:
        return True

    if board[pos] != 0:
        return _fill_complete_board(board, rng, pos + 1)

    values = list(range(1, GRID_SIZE + 1))
    rng.shuffle(values)
    for val in values:
        if is_valid_placement(board, pos, val):
            board[pos] = val
            if _fill_complete_board(board, rng, pos + 1):
                return True
            board[pos] = 0

    return False


def generate_sudoku(
    num_empty: int = 45,
    seed: int | None = None,
    *,
    ensure_unique: bool = True,
    max_attempts: int = 20,
) -> tuple[list[int], list[int]]:
    """
    Generate a valid Sudoku puzzle and its solution.

    When ``ensure_unique`` is true, removals are accepted only if the puzzle has
    exactly one solution. If the requested target cannot be reached on the first
    solved board, the function retries with deterministic derived seeds.
    """
    target_empty = max(0, min(num_empty, BOARD_SIZE - 17))

    for attempt in range(max_attempts):
        attempt_seed = None if seed is None else seed + attempt * 9973
        rng = _rng(attempt_seed)
        solved = [0] * BOARD_SIZE
        if not _fill_complete_board(solved, rng):
            continue

        puzzle = solved[:]
        cells = list(range(BOARD_SIZE))
        rng.shuffle(cells)
        removed = 0

        for idx in cells:
            if removed >= target_empty:
                break

            previous = puzzle[idx]
            puzzle[idx] = 0
            if ensure_unique and count_solutions(puzzle, limit=2) != 1:
                puzzle[idx] = previous
                continue
            removed += 1

        if removed >= target_empty or not ensure_unique:
            return puzzle, solved

    raise RuntimeError(f"Could not generate a unique Sudoku puzzle with {target_empty} empty cells")
