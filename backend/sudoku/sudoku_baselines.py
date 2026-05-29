"""
Sudoku baseline solvers used for comparative evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from sudoku.sudoku_utils import solve_sudoku_board


@dataclass(frozen=True)
class SudokuBaselineResult:
    method: str
    solved: bool
    steps: int
    duration_ms: float
    board: list[int]

    def to_summary(self) -> dict[str, int | float | str | bool]:
        return {
            "method": self.method,
            "solved": self.solved,
            "steps": self.steps,
            "duration_ms": round(self.duration_ms, 3),
        }


def solve_backtracking(puzzle: list[int], *, max_nodes: int = 100_000) -> SudokuBaselineResult:
    started = time.perf_counter()
    board, solved, steps = solve_sudoku_board(
        puzzle,
        use_mrv=False,
        randomize=False,
        max_nodes=max_nodes,
    )
    return SudokuBaselineResult(
        method="Pure Backtracking",
        solved=solved,
        steps=steps,
        duration_ms=(time.perf_counter() - started) * 1000,
        board=board,
    )


def solve_mrv(puzzle: list[int], *, max_nodes: int = 100_000) -> SudokuBaselineResult:
    started = time.perf_counter()
    board, solved, steps = solve_sudoku_board(
        puzzle,
        use_mrv=True,
        randomize=False,
        max_nodes=max_nodes,
    )
    return SudokuBaselineResult(
        method="MRV Heuristic",
        solved=solved,
        steps=steps,
        duration_ms=(time.perf_counter() - started) * 1000,
        board=board,
    )


def solve_random_mrv(
    puzzle: list[int],
    *,
    seed: int | None = None,
    max_nodes: int = 100_000,
) -> SudokuBaselineResult:
    started = time.perf_counter()
    board, solved, steps = solve_sudoku_board(
        puzzle,
        use_mrv=True,
        randomize=True,
        seed=seed,
        max_nodes=max_nodes,
    )
    return SudokuBaselineResult(
        method="Random MRV",
        solved=solved,
        steps=steps,
        duration_ms=(time.perf_counter() - started) * 1000,
        board=board,
    )
