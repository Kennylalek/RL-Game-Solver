"""
MRV-guided tabular Q-learning for Sudoku.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import pickle
import random
from typing import Any, Callable

from sudoku.sudoku_utils import (
    empty_count_for_difficulty,
    generate_sudoku,
    get_candidates,
    mrv_cell,
    state_features,
)

DEFAULT_AGENT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "agents"


class SudokuQLearner:
    def __init__(self, difficulty: str = "medium", seed: int | None = None):
        self.diff = difficulty
        self.num_empty = empty_count_for_difficulty(difficulty)
        self.q: defaultdict[tuple[tuple[int, int, int, int], int, int], float] = defaultdict(float)
        self.alpha = 0.3
        self.gamma = 0.95
        self.epsilon = 1.0
        self.eps_decay = 0.995
        self.eps_min = 0.05
        self.trained = False
        self.rng = random.Random(seed)
        self.training_history: list[dict[str, float | int | bool]] = []

    def choose_action(self, board: list[int], *, explore: bool = True) -> tuple[int, int] | None:
        """Choose a digit for the MRV cell with epsilon-greedy exploration."""
        idx, candidates = mrv_cell(board)
        if idx == -1 or not candidates:
            return None

        if explore and self.rng.random() < self.epsilon:
            return idx, self.rng.choice(candidates)

        values = self.action_values(board)
        best = max(values["candidates"], key=lambda item: (item["q"], -item["value"]))
        return idx, int(best["value"])

    def action_values(self, board: list[int]) -> dict[str, Any]:
        idx, candidates = mrv_cell(board)
        features = state_features(board)
        return {
            "state": features,
            "cell": idx,
            "row": idx // 9 if idx >= 0 else None,
            "col": idx % 9 if idx >= 0 else None,
            "candidates": [
                {"value": val, "q": round(self.q[(features, idx, val)], 6)}
                for val in candidates
            ],
        }

    def _max_next_q(self, next_board: list[int]) -> float:
        next_idx, next_candidates = mrv_cell(next_board)
        if next_idx == -1 or not next_candidates:
            return 0.0

        next_state = state_features(next_board)
        return max(self.q[(next_state, next_idx, val)] for val in next_candidates)

    def update(
        self,
        state: tuple[int, int, int, int],
        action: tuple[int, int] | None,
        reward: float,
        next_board: list[int],
        done: bool,
    ) -> None:
        if action is None:
            return

        idx, val = action
        key = (state, idx, val)
        target = reward if done else reward + self.gamma * self._max_next_q(next_board)
        self.q[key] += self.alpha * (target - self.q[key])

    def train(
        self,
        episodes: int = 300,
        *,
        unique_puzzles: bool = False,
        progress_callback: Callable[[dict[str, float | int | bool]], None] | None = None,
    ) -> list[dict[str, float | int | bool]]:
        self.training_history = []

        for episode in range(1, episodes + 1):
            puzzle_seed = self.rng.randint(0, 2_000_000_000)
            puzzle, _ = generate_sudoku(self.num_empty, seed=puzzle_seed, ensure_unique=unique_puzzles)
            board = puzzle[:]
            episode_reward = 0.0
            solved = False
            conflict = False

            for step in range(1, 201):
                state = state_features(board)
                action = self.choose_action(board)
                if action is None:
                    break

                idx, val = action
                board[idx] = val
                empty = board.count(0)

                if empty == 0:
                    reward = 10.0
                    done = True
                    solved = True
                else:
                    conflict = any(
                        len(get_candidates(board, cell)) == 0
                        for cell in range(81)
                        if board[cell] == 0
                    )
                    reward = -5.0 if conflict else 0.2
                    done = conflict

                episode_reward += reward
                self.update(state, action, reward, board, done)

                if conflict:
                    board[idx] = 0
                    break
                if done:
                    break

            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
            self.training_history.append({
                "episode": episode,
                "reward": round(episode_reward, 6),
                "solved": solved,
                "conflict": conflict,
                "filled": 81 - board.count(0),
                "epsilon": round(self.epsilon, 6),
                "q_size": len(self.q),
            })
            if progress_callback is not None:
                progress_callback(self.training_history[-1])

        self.trained = True
        return self.training_history

    def solve_with_steps(
        self,
        puzzle: list[int],
        *,
        max_steps: int = 5000,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Produce a step-by-step trace using the trained policy plus backtracking."""
        board = puzzle[:]
        steps: list[dict[str, Any]] = [{
            "board": board[:],
            "action": None,
            "message": "Puzzle loaded",
            "reward": 0,
            "explain": self.action_values(board),
        }]
        if progress_callback is not None:
            progress_callback(steps[-1])

        stack: list[tuple[list[int], int, list[int]]] = []

        for _ in range(max_steps):
            idx, candidates = mrv_cell(board)

            if idx == -1:
                steps.append({
                    "board": board[:],
                    "action": None,
                    "message": "Solved",
                    "reward": 10,
                    "explain": {"state": state_features(board), "cell": None, "candidates": []},
                })
                if progress_callback is not None:
                    progress_callback(steps[-1])
                return steps, True

            if not candidates:
                if not stack:
                    break
                board, stack_idx, remaining = stack.pop()
                if not remaining:
                    continue
                val = remaining.pop()
                board[stack_idx] = val
                steps.append({
                    "board": board[:],
                    "action": {"idx": stack_idx, "val": val},
                    "message": f"Backtrack: cell {stack_idx} = {val}",
                    "reward": -1,
                    "explain": self.action_values(board),
                })
                if progress_callback is not None:
                    progress_callback(steps[-1])
                stack.append((board[:], stack_idx, remaining))
                continue

            explanation = self.action_values(board)
            ranked = sorted(
                explanation["candidates"],
                key=lambda item: (float(item["q"]), -int(item["value"])),
                reverse=True,
            )
            best_val = int(ranked[0]["value"])
            best_q = float(ranked[0]["q"])

            remaining = [val for val in candidates if val != best_val]
            remaining.sort(
                key=lambda val: self.q[(state_features(board), idx, val)],
                reverse=True,
            )
            stack.append((board[:], idx, remaining))
            board[idx] = best_val

            steps.append({
                "board": board[:],
                "action": {"idx": idx, "val": best_val},
                "message": f"Place {best_val} at ({idx // 9}, {idx % 9}) | Q={best_q:.2f}",
                "reward": round(best_q, 2),
                "explain": explanation,
            })
            if progress_callback is not None:
                progress_callback(steps[-1])

        steps.append({
            "board": board[:],
            "action": None,
            "message": "Partial solution; increase training or use baseline backtracking",
            "reward": 0,
            "explain": self.action_values(board),
        })
        if progress_callback is not None:
            progress_callback(steps[-1])
        return steps, board.count(0) == 0

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else DEFAULT_AGENT_DIR / f"sudoku_{self.diff}.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "difficulty": self.diff,
            "q": dict(self.q),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "eps_decay": self.eps_decay,
            "eps_min": self.eps_min,
            "trained": self.trained,
            "training_history": self.training_history,
        }
        with target.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SudokuQLearner":
        source = Path(path)
        with source.open("rb") as handle:
            payload = pickle.load(handle)

        agent = cls(difficulty=payload["difficulty"])
        agent.q = defaultdict(float, payload["q"])
        agent.alpha = payload["alpha"]
        agent.gamma = payload["gamma"]
        agent.epsilon = payload["epsilon"]
        agent.eps_decay = payload["eps_decay"]
        agent.eps_min = payload["eps_min"]
        agent.trained = payload["trained"]
        agent.training_history = payload.get("training_history", [])
        return agent
