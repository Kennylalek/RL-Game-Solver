"""
Tabular Q-learning for 2048 with deterministic evaluation policies.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import pickle
import random
from typing import Any, Callable

from game2048.game2048_utils import (
    ACTION_NAMES,
    ACTIONS_2048,
    add_tile,
    board_features,
    clone_board,
    empty_count,
    heuristic_value,
    is_game_over_2048,
    max_tile,
    move_2048,
    new_board_2048,
    valid_actions_2048,
)

DEFAULT_AGENT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "agents"


class Game2048QLearner:
    def __init__(self, seed: int | None = None):
        self.q: defaultdict[tuple[tuple[int, int, int, int, int], int], float] = defaultdict(float)
        self.alpha = 0.2
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_decay = 0.997
        self.eps_min = 0.05
        self.trained = False
        self.best_score = 0
        self.rng = random.Random(seed)
        self.training_history: list[dict[str, float | int]] = []

    def action_values(self, board: list[list[int]]) -> dict[str, Any]:
        features = board_features(board)
        valid = valid_actions_2048(board)
        values = []
        for action in ACTIONS_2048:
            moved_board, score_gain, moved = move_2048(board, action)
            values.append({
                "action": action,
                "name": ACTION_NAMES[action],
                "valid": moved,
                "q": round(self.q[(features, action)], 6),
                "score_gain": score_gain,
                "heuristic": round(heuristic_value(moved_board), 3) if moved else None,
            })
        return {"state": features, "valid_actions": valid, "actions": values}

    def choose_action(self, board: list[list[int]], *, explore: bool = True) -> int | None:
        valid = valid_actions_2048(board)
        if not valid:
            return None

        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(valid)

        return self.q_action(board)

    def q_action(self, board: list[list[int]]) -> int | None:
        valid = valid_actions_2048(board)
        if not valid:
            return None

        state = board_features(board)
        return max(valid, key=lambda action: (self.q[(state, action)], action))

    def update(
        self,
        state: tuple[int, int, int, int, int],
        action: int | None,
        reward: float,
        next_board: list[list[int]],
        done: bool,
    ) -> None:
        if action is None:
            return

        key = (state, action)
        if done:
            target = reward
        else:
            next_state = board_features(next_board)
            valid_next = valid_actions_2048(next_board)
            future = max((self.q[(next_state, candidate)] for candidate in valid_next), default=0.0)
            target = reward + self.gamma * future

        self.q[key] += self.alpha * (target - self.q[key])

    def reward_shaping(
        self,
        board: list[list[int]],
        new_board: list[list[int]],
        score_gain: int,
    ) -> float:
        merge_reward = math.log2(score_gain + 1) * 10
        empty_bonus = empty_count(new_board) * 2
        max_bonus = math.log2(max_tile(new_board)) * 0.5 if max_tile(new_board) > 0 else 0
        heuristic_delta = (heuristic_value(new_board) - heuristic_value(board)) * 0.02
        return merge_reward + empty_bonus + max_bonus + heuristic_delta

    def train(
        self,
        episodes: int = 1000,
        *,
        progress_callback: Callable[[dict[str, float | int]], None] | None = None,
    ) -> list[dict[str, float | int]]:
        self.training_history = []

        for episode in range(1, episodes + 1):
            board = new_board_2048(rng=self.rng)
            episode_score = 0
            moves = 0

            for _ in range(2000):
                state = board_features(board)
                action = self.choose_action(board)
                if action is None:
                    break

                new_board, score_gain, moved = move_2048(board, action)
                if not moved:
                    self.update(state, action, -5.0, board, False)
                    continue

                episode_score += score_gain
                moves += 1
                add_tile(new_board, rng=self.rng)
                done = is_game_over_2048(new_board)
                reward = self.reward_shaping(board, new_board, score_gain)
                if done:
                    reward -= 20.0

                self.update(state, action, reward, new_board, done)
                board = new_board
                if done:
                    break

            self.best_score = max(self.best_score, episode_score)
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
            self.training_history.append({
                "episode": episode,
                "score": episode_score,
                "moves": moves,
                "max_tile": max_tile(board),
                "epsilon": round(self.epsilon, 6),
                "q_size": len(self.q),
            })
            if progress_callback is not None:
                progress_callback(self.training_history[-1])

        self.trained = True
        return self.training_history

    def play_episode(
        self,
        *,
        policy: str = "hybrid",
        seed: int | None = None,
        max_moves: int = 2000,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        board = new_board_2048(rng=rng)
        steps: list[dict[str, Any]] = []
        total_score = 0

        steps.append({
            "board": clone_board(board),
            "action": None,
            "action_name": None,
            "score": 0,
            "reward": 0,
            "message": "Game started",
            "policy": policy,
            "explain": self.action_values(board),
        })
        if progress_callback is not None:
            progress_callback(steps[-1])

        for move_num in range(1, max_moves + 1):
            action = self.policy_action(board, policy=policy, rng=rng)
            if action is None:
                break

            new_board, score_gain, moved = move_2048(board, action)
            if not moved:
                break

            total_score += score_gain
            reward = self.reward_shaping(board, new_board, score_gain)
            add_tile(new_board, rng=rng)
            done = is_game_over_2048(new_board)

            steps.append({
                "board": clone_board(new_board),
                "action": action,
                "action_name": ACTION_NAMES[action],
                "score": total_score,
                "reward": round(reward, 3),
                "message": f"Move {move_num}: {ACTION_NAMES[action]} | Score: {total_score}",
                "policy": policy,
                "explain": self.action_values(board),
            })

            board = new_board
            if done:
                steps[-1]["message"] = (
                    f"Game Over | Score: {total_score} | Max tile: {max_tile(board)}"
                )
            if progress_callback is not None:
                progress_callback(steps[-1])
            if done:
                break

        return steps

    def policy_action(
        self,
        board: list[list[int]],
        *,
        policy: str,
        rng: random.Random | None = None,
    ) -> int | None:
        if policy == "q":
            return self.q_action(board)
        if policy == "expectimax":
            return self.expectimax_action(board, depth=2, use_q_leaf=False)
        if policy == "random":
            valid = valid_actions_2048(board)
            source = rng if rng is not None else self.rng
            return source.choice(valid) if valid else None
        if policy == "hybrid":
            return self.expectimax_action(board, depth=2, use_q_leaf=True)
        raise ValueError(f"Unknown 2048 policy: {policy}")

    def expectimax_action(
        self,
        board: list[list[int]],
        *,
        depth: int = 2,
        use_q_leaf: bool = False,
    ) -> int | None:
        valid = valid_actions_2048(board)
        if not valid:
            return None

        return max(
            valid,
            key=lambda action: self._action_expectimax_value(
                board,
                action,
                depth=depth,
                use_q_leaf=use_q_leaf,
            ),
        )

    def _action_expectimax_value(
        self,
        board: list[list[int]],
        action: int,
        *,
        depth: int,
        use_q_leaf: bool,
    ) -> float:
        new_board, score, moved = move_2048(board, action)
        if not moved:
            return -1e9
        return score + self._expect(new_board, depth - 1, use_q_leaf=use_q_leaf)

    def _expect(self, board: list[list[int]], depth: int, *, use_q_leaf: bool) -> float:
        if depth <= 0:
            value = heuristic_value(board)
            if use_q_leaf:
                value += self._max_q_value(board) * 0.25
            return value

        empties = [(row, col) for row in range(4) for col in range(4) if board[row][col] == 0]
        if not empties:
            return self._max_val(board, depth, use_q_leaf=use_q_leaf)

        sample = empties[:4]
        total = 0.0
        for row, col in sample:
            for value, probability in ((2, 0.9), (4, 0.1)):
                next_board = clone_board(board)
                next_board[row][col] = value
                total += probability * self._max_val(next_board, depth - 1, use_q_leaf=use_q_leaf)
        return total / len(sample)

    def _max_val(self, board: list[list[int]], depth: int, *, use_q_leaf: bool) -> float:
        valid = valid_actions_2048(board)
        if not valid:
            return heuristic_value(board)
        if depth <= 0:
            value = heuristic_value(board)
            if use_q_leaf:
                value += self._max_q_value(board) * 0.25
            return value
        return max(
            self._action_expectimax_value(board, action, depth=depth, use_q_leaf=use_q_leaf)
            for action in valid
        )

    def _max_q_value(self, board: list[list[int]]) -> float:
        state = board_features(board)
        valid = valid_actions_2048(board)
        return max((self.q[(state, action)] for action in valid), default=0.0)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else DEFAULT_AGENT_DIR / "game2048.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q": dict(self.q),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "eps_decay": self.eps_decay,
            "eps_min": self.eps_min,
            "trained": self.trained,
            "best_score": self.best_score,
            "training_history": self.training_history,
        }
        with target.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Game2048QLearner":
        source = Path(path)
        with source.open("rb") as handle:
            payload = pickle.load(handle)

        agent = cls()
        agent.q = defaultdict(float, payload["q"])
        agent.alpha = payload["alpha"]
        agent.gamma = payload["gamma"]
        agent.epsilon = payload["epsilon"]
        agent.eps_decay = payload["eps_decay"]
        agent.eps_min = payload["eps_min"]
        agent.trained = payload["trained"]
        agent.best_score = payload["best_score"]
        agent.training_history = payload.get("training_history", [])
        return agent
