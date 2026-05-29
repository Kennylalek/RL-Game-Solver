"""
RL Game Solver Flask API.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request
from flask_cors import CORS

from analysis import run_comparison, write_experiment_artifacts
from game2048.game2048_rl import DEFAULT_AGENT_DIR as GAME_AGENT_DIR
from game2048.game2048_rl import Game2048QLearner
from sudoku.sudoku_baselines import solve_backtracking, solve_mrv, solve_random_mrv
from sudoku.sudoku_rl import DEFAULT_AGENT_DIR as SUDOKU_AGENT_DIR
from sudoku.sudoku_rl import SudokuQLearner
from sudoku.sudoku_utils import count_solutions, empty_count_for_difficulty, generate_sudoku

app = Flask(__name__)
CORS(app)

training_status = {
    "sudoku": {"easy": "idle", "medium": "idle", "hard": "idle"},
    "2048": "idle",
}
agents = {
    "sudoku": {},
    "2048": None,
}
live_jobs: dict[str, dict] = {}
state_lock = threading.Lock()
ROOT_DIR = Path(__file__).resolve().parents[1]
SNAKE_WEIGHTS_2048 = [
    15, 14, 13, 12,
    8, 9, 10, 11,
    7, 6, 5, 4,
    0, 1, 2, 3,
]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _create_job(
    *,
    domain: str,
    kind: str,
    total: int | None,
    message: str,
    metadata: dict | None = None,
) -> dict:
    job_id = uuid4().hex
    job = {
        "id": job_id,
        "domain": domain,
        "kind": kind,
        "status": "queued",
        "progress": 0,
        "total": total,
        "message": message,
        "history": [],
        "steps": [],
        "result": None,
        "error": None,
        "metadata": metadata or {},
        "created_at": _now_ms(),
        "updated_at": _now_ms(),
    }
    with state_lock:
        live_jobs[job_id] = job
    return job


def _patch_job(job_id: str, **fields) -> None:
    with state_lock:
        job = live_jobs[job_id]
        job.update(fields)
        job["updated_at"] = _now_ms()


def _append_job_history(job_id: str, row: dict, total: int) -> None:
    with state_lock:
        job = live_jobs[job_id]
        row_with_metrics = _row_with_rolling_metrics(job["history"], row)
        job["history"].append(row_with_metrics)
        job["progress"] = int(row_with_metrics.get("episode", len(job["history"])))
        job["total"] = total
        job["message"] = f"Episode {job['progress']} / {total}"
        job["updated_at"] = _now_ms()


def _append_job_step(job_id: str, step: dict) -> None:
    with state_lock:
        job = live_jobs[job_id]
        job["steps"].append(step)
        job["progress"] = len(job["steps"])
        job["message"] = step.get("message", f"Step {job['progress']}")
        job["updated_at"] = _now_ms()


def _finish_job(job_id: str, *, result: dict | None = None, message: str = "Complete") -> None:
    _patch_job(job_id, status="complete", result=result or {}, message=message)


def _fail_job(job_id: str, error: Exception) -> None:
    _patch_job(job_id, status="error", error=str(error), message=str(error))


def _job_snapshot(job_id: str) -> dict | None:
    with state_lock:
        job = live_jobs.get(job_id)
        if job is None:
            return None
        return {
            **job,
            "history": list(job["history"]),
            "steps": list(job["steps"]),
            "metadata": dict(job["metadata"]),
        }


def _load_persisted_agents() -> None:
    for difficulty in ("easy", "medium", "hard"):
        path = SUDOKU_AGENT_DIR / f"sudoku_{difficulty}.pkl"
        if path.exists():
            try:
                agents["sudoku"][difficulty] = SudokuQLearner.load(path)
                training_status["sudoku"][difficulty] = "ready"
            except Exception:
                training_status["sudoku"][difficulty] = "idle"

    path = GAME_AGENT_DIR / "game2048.pkl"
    if path.exists():
        try:
            agents["2048"] = Game2048QLearner.load(path)
            training_status["2048"] = "ready"
        except Exception:
            training_status["2048"] = "idle"


def _json_payload() -> dict:
    return request.get_json(silent=True) or {}


def _optional_int(value, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    return int(value)


def _clamped_float(payload: dict, key: str, default: float, minimum: float, maximum: float) -> float:
    if key not in payload or payload[key] in (None, ""):
        return default
    value = float(payload[key])
    return max(minimum, min(maximum, value))


def _apply_agent_hyperparameters(agent, payload: dict) -> dict[str, float]:
    agent.alpha = _clamped_float(payload, "alpha", agent.alpha, 0.001, 1.0)
    agent.gamma = _clamped_float(payload, "gamma", agent.gamma, 0.0, 1.0)
    agent.epsilon = _clamped_float(payload, "epsilon", agent.epsilon, 0.0, 1.0)
    agent.eps_decay = _clamped_float(payload, "eps_decay", agent.eps_decay, 0.5, 1.0)
    agent.eps_min = _clamped_float(payload, "eps_min", agent.eps_min, 0.0, 1.0)
    agent.eps_min = min(agent.eps_min, agent.epsilon)
    return {
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "epsilon": agent.epsilon,
        "eps_decay": agent.eps_decay,
        "eps_min": agent.eps_min,
    }


def _row_with_rolling_metrics(previous_rows: list[dict], row: dict, window: int = 25) -> dict:
    enriched = dict(row)
    metric_keys = ("reward", "filled", "score", "max_tile", "moves")
    recent_rows = previous_rows[-(window - 1):] + [enriched]
    for key in metric_keys:
        if key not in enriched:
            continue
        values = [
            float(item[key])
            for item in recent_rows
            if item.get(key) is not None and isinstance(item.get(key), (int, float))
        ]
        if values:
            enriched[f"rolling_{key}"] = round(sum(values) / len(values), 6)
    return enriched


def _history_with_rolling(history: list[dict], window: int = 25) -> list[dict]:
    enriched: list[dict] = []
    for row in history:
        enriched.append(_row_with_rolling_metrics(enriched, row, window=window))
    return enriched


def _sudoku_validation_summary(puzzle: list[int], difficulty: str) -> dict:
    solution_count = count_solutions(puzzle, limit=2)
    mrv_result = solve_mrv(puzzle)
    backtracking_result = solve_backtracking(puzzle)
    return {
        "difficulty": difficulty,
        "target_empty_cells": empty_count_for_difficulty(difficulty),
        "actual_empty_cells": puzzle.count(0),
        "unique_solution": solution_count == 1,
        "solution_count_capped": solution_count,
        "mrv": mrv_result.to_summary(),
        "backtracking": backtracking_result.to_summary(),
    }


def _parse_comparison_seeds(payload: dict) -> list[int]:
    seeds_raw = payload.get("seeds")
    if seeds_raw:
        if isinstance(seeds_raw, str):
            return [int(seed.strip()) for seed in seeds_raw.split(",") if seed.strip()]
        return [int(seed) for seed in seeds_raw]

    runs = int(payload.get("runs") or request.args.get("runs", 3))
    seed_start = int(payload.get("seed") or request.args.get("seed", 101))
    return [seed_start + 101 * idx for idx in range(max(1, min(runs, 10)))]


def _sudoku_baseline_steps(
    puzzle: list[int],
    algorithm: str,
    *,
    seed: int | None = None,
) -> tuple[list[dict], bool]:
    if algorithm == "mrv":
        result = solve_mrv(puzzle)
        method = "MRV Heuristic"
    elif algorithm == "backtracking":
        result = solve_backtracking(puzzle)
        method = "Pure Backtracking"
    elif algorithm == "random_mrv":
        result = solve_random_mrv(puzzle, seed=seed)
        method = "Random MRV"
    else:
        raise ValueError(f"Unknown Sudoku algorithm: {algorithm}")

    initial = {
        "board": puzzle[:],
        "action": None,
        "message": f"{method} started",
        "reward": 0,
        "explain": {
            "state": [puzzle.count(0), result.steps, 0, 0],
            "cell": None,
            "row": None,
            "col": None,
            "candidates": [],
            "algorithm": method,
        },
    }
    final = {
        "board": result.board[:],
        "action": None,
        "message": f"{method} {'solved' if result.solved else 'stopped'} in {result.steps} steps",
        "reward": 10 if result.solved else 0,
        "explain": {
            "state": [result.board.count(0), result.steps, round(result.duration_ms, 3), 0],
            "cell": None,
            "row": None,
            "col": None,
            "candidates": [],
            "algorithm": method,
            "duration_ms": round(result.duration_ms, 3),
        },
    }
    return [initial, final], result.solved


def _solve_sudoku_with_algorithm(
    *,
    difficulty: str,
    puzzle: list[int],
    algorithm: str,
    seed: int | None = None,
    progress_callback=None,
) -> tuple[list[dict], bool, list[dict]]:
    if algorithm == "rl":
        agent = agents["sudoku"].get(difficulty)
        if not agent or not agent.trained:
            raise RuntimeError("RL agent not trained")
        steps, solved = agent.solve_with_steps(puzzle, progress_callback=progress_callback)
        return steps, solved, agent.training_history

    steps, solved = _sudoku_baseline_steps(puzzle, algorithm, seed=seed)
    if progress_callback is not None:
        for step in steps:
            progress_callback(step)
    return steps, solved, []


def _get_or_train_sudoku_for_analysis(difficulty: str) -> SudokuQLearner:
    agent = agents["sudoku"].get(difficulty)
    if agent and agent.trained:
        return agent

    agent = SudokuQLearner(difficulty=difficulty, seed=29)
    agent.train(episodes=80)
    agent.save()
    agents["sudoku"][difficulty] = agent
    training_status["sudoku"][difficulty] = "ready"
    return agent


def _get_or_train_2048_for_analysis() -> Game2048QLearner:
    agent = agents["2048"]
    if agent and agent.trained:
        return agent

    agent = Game2048QLearner(seed=29)
    agent.train(episodes=160)
    agent.save()
    agents["2048"] = agent
    training_status["2048"] = "ready"
    return agent


def _get_game_agent_for_policy(policy: str, seed: int | None) -> Game2048QLearner:
    agent = agents["2048"]
    if agent and agent.trained:
        return agent

    if policy in {"expectimax", "random"}:
        return Game2048QLearner(seed=seed)

    raise RuntimeError("2048 RL agent not trained")


@app.route("/api/status")
def status():
    return jsonify({
        "training": {
            "sudoku": training_status["sudoku"],
            "2048": training_status["2048"],
        },
        "agents": {
            "sudoku": sorted(agents["sudoku"].keys()),
            "2048": agents["2048"] is not None,
        },
    })


@app.route("/api/jobs/<job_id>")
def get_job(job_id: str):
    job = _job_snapshot(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/sudoku/model", methods=["POST"])
def sudoku_model():
    payload = _json_payload()
    difficulty = payload.get("difficulty", "medium")
    action = payload.get("action", "save")
    path = SUDOKU_AGENT_DIR / f"sudoku_{difficulty}.pkl"

    if action == "save":
        agent = agents["sudoku"].get(difficulty)
        if not agent or not agent.trained:
            return jsonify({"error": "Sudoku agent not trained"}), 400
        saved_path = agent.save(path)
        return jsonify({
            "status": "saved",
            "message": f"Saved Sudoku {difficulty} model",
            "path": str(saved_path),
        })

    if action == "load":
        if not path.exists():
            return jsonify({"error": f"No saved Sudoku {difficulty} model found"}), 404
        agent = SudokuQLearner.load(path)
        with state_lock:
            agents["sudoku"][difficulty] = agent
            training_status["sudoku"][difficulty] = "ready" if agent.trained else "idle"
        return jsonify({
            "status": "loaded",
            "message": f"Loaded Sudoku {difficulty} model",
            "path": str(path),
            "episodes": len(agent.training_history),
            "q_size": len(agent.q),
        })

    if action == "reset":
        with state_lock:
            agents["sudoku"].pop(difficulty, None)
            training_status["sudoku"][difficulty] = "idle"
        path.unlink(missing_ok=True)
        return jsonify({
            "status": "reset",
            "message": f"Reset Sudoku {difficulty} model",
            "path": str(path),
        })

    return jsonify({"error": f"Unknown model action: {action}"}), 400


@app.route("/api/2048/model", methods=["POST"])
def game2048_model():
    payload = _json_payload()
    action = payload.get("action", "save")
    path = GAME_AGENT_DIR / "game2048.pkl"

    if action == "save":
        agent = agents["2048"]
        if not agent or not agent.trained:
            return jsonify({"error": "2048 agent not trained"}), 400
        saved_path = agent.save(path)
        return jsonify({
            "status": "saved",
            "message": "Saved 2048 model",
            "path": str(saved_path),
        })

    if action == "load":
        if not path.exists():
            return jsonify({"error": "No saved 2048 model found"}), 404
        agent = Game2048QLearner.load(path)
        with state_lock:
            agents["2048"] = agent
            training_status["2048"] = "ready" if agent.trained else "idle"
        return jsonify({
            "status": "loaded",
            "message": "Loaded 2048 model",
            "path": str(path),
            "episodes": len(agent.training_history),
            "q_size": len(agent.q),
            "best_score": agent.best_score,
        })

    if action == "reset":
        with state_lock:
            agents["2048"] = None
            training_status["2048"] = "idle"
        path.unlink(missing_ok=True)
        return jsonify({
            "status": "reset",
            "message": "Reset 2048 model",
            "path": str(path),
        })

    return jsonify({"error": f"Unknown model action: {action}"}), 400


@app.route("/api/sudoku/train", methods=["POST"])
def sudoku_train():
    payload = _json_payload()
    difficulty = payload.get("difficulty", "medium")
    episodes = int(payload.get("episodes", {"easy": 200, "medium": 300, "hard": 400}.get(difficulty, 300)))
    seed = _optional_int(payload.get("seed"), episodes)
    job = _create_job(
        domain="sudoku",
        kind="training",
        total=episodes,
        message=f"Queued Sudoku {difficulty} training",
        metadata={"difficulty": difficulty, "episodes": episodes, "seed": seed},
    )

    def train_worker() -> None:
        try:
            _patch_job(job["id"], status="running", message=f"Training Sudoku {difficulty}")
            with state_lock:
                training_status["sudoku"][difficulty] = "training"
            agent = SudokuQLearner(difficulty=difficulty, seed=seed)
            hyperparameters = _apply_agent_hyperparameters(agent, payload)
            _patch_job(job["id"], metadata={
                "difficulty": difficulty,
                "episodes": episodes,
                "seed": seed,
                "hyperparameters": hyperparameters,
            })
            agent.train(
                episodes=episodes,
                progress_callback=lambda row: _append_job_history(job["id"], row, episodes),
            )
            agent.save()
            with state_lock:
                agents["sudoku"][difficulty] = agent
                training_status["sudoku"][difficulty] = "ready"
            _finish_job(
                job["id"],
                result={
                    "difficulty": difficulty,
                    "episodes": episodes,
                    "q_size": len(agent.q),
                    "seed": seed,
                    "hyperparameters": hyperparameters,
                    "training_history": _history_with_rolling(agent.training_history),
                },
                message=f"Sudoku {difficulty} training complete",
            )
        except Exception as exc:
            with state_lock:
                training_status["sudoku"][difficulty] = "idle"
            _fail_job(job["id"], exc)

    threading.Thread(target=train_worker, daemon=True).start()
    return jsonify({
        "status": "training started",
        "difficulty": difficulty,
        "episodes": episodes,
        "seed": seed,
        "job_id": job["id"],
    })


@app.route("/api/sudoku/puzzle")
def sudoku_puzzle():
    difficulty = request.args.get("difficulty", "medium")
    seed_arg = request.args.get("seed")
    seed = int(seed_arg) if seed_arg is not None else None
    num_empty = empty_count_for_difficulty(difficulty)
    puzzle, solution = generate_sudoku(num_empty, seed=seed, ensure_unique=True)
    validation = _sudoku_validation_summary(puzzle, difficulty)
    return jsonify({
        "board": puzzle,
        "solution": solution,
        "difficulty": difficulty,
        "empty_cells": num_empty,
        "unique_solution": True,
        "validation": validation,
    })


@app.route("/api/sudoku/solve", methods=["POST"])
def sudoku_solve():
    payload = _json_payload()
    difficulty = payload.get("difficulty", "medium")
    algorithm = payload.get("algorithm", "rl")
    seed = _optional_int(payload.get("seed"))

    puzzle = payload.get("board")
    if not puzzle or len(puzzle) != 81:
        puzzle, _ = generate_sudoku(
            empty_count_for_difficulty(difficulty),
            seed=seed,
            ensure_unique=True,
        )

    if payload.get("live"):
        job = _create_job(
            domain="sudoku",
            kind="solve",
            total=None,
            message=f"Queued Sudoku {algorithm} solve",
            metadata={"difficulty": difficulty, "algorithm": algorithm},
        )

        def solve_worker() -> None:
            try:
                _patch_job(job["id"], status="running", message=f"Solving Sudoku with {algorithm}")
                steps, solved, training_history = _solve_sudoku_with_algorithm(
                    difficulty=difficulty,
                    puzzle=puzzle,
                    algorithm=algorithm,
                    seed=seed,
                    progress_callback=lambda step: _append_job_step(job["id"], step),
                )
                _finish_job(
                    job["id"],
                    result={
                        "puzzle": puzzle,
                        "algorithm": algorithm,
                        "solved": solved,
                        "total_steps": len(steps),
                        "training_history": _history_with_rolling(training_history),
                    },
                    message="Sudoku solved" if solved else "Sudoku partial solution",
                )
            except Exception as exc:
                _fail_job(job["id"], exc)

        threading.Thread(target=solve_worker, daemon=True).start()
        return jsonify({"status": "solve started", "job_id": job["id"], "algorithm": algorithm})

    try:
        steps, solved, training_history = _solve_sudoku_with_algorithm(
            difficulty=difficulty,
            puzzle=puzzle,
            algorithm=algorithm,
            seed=seed,
        )
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "puzzle": puzzle,
        "algorithm": algorithm,
        "steps": steps,
        "solved": solved,
        "total_steps": len(steps),
        "training_history": _history_with_rolling(training_history),
        "validation": _sudoku_validation_summary(puzzle, difficulty),
    })


@app.route("/api/2048/train", methods=["POST"])
def game2048_train():
    payload = _json_payload()
    episodes = int(payload.get("episodes", 1000))
    seed = _optional_int(payload.get("seed"), episodes)
    job = _create_job(
        domain="2048",
        kind="training",
        total=episodes,
        message="Queued 2048 training",
        metadata={"episodes": episodes, "seed": seed},
    )

    def train_worker() -> None:
        try:
            _patch_job(job["id"], status="running", message="Training 2048")
            with state_lock:
                training_status["2048"] = "training"
            agent = Game2048QLearner(seed=seed)
            hyperparameters = _apply_agent_hyperparameters(agent, payload)
            _patch_job(job["id"], metadata={
                "episodes": episodes,
                "seed": seed,
                "hyperparameters": hyperparameters,
            })
            agent.train(
                episodes=episodes,
                progress_callback=lambda row: _append_job_history(job["id"], row, episodes),
            )
            agent.save()
            with state_lock:
                agents["2048"] = agent
                training_status["2048"] = "ready"
            _finish_job(
                job["id"],
                result={
                    "episodes": episodes,
                    "q_size": len(agent.q),
                    "best_score": agent.best_score,
                    "seed": seed,
                    "hyperparameters": hyperparameters,
                    "training_history": _history_with_rolling(agent.training_history),
                },
                message="2048 training complete",
            )
        except Exception as exc:
            with state_lock:
                training_status["2048"] = "idle"
            _fail_job(job["id"], exc)

    threading.Thread(target=train_worker, daemon=True).start()
    return jsonify({"status": "training started", "episodes": episodes, "seed": seed, "job_id": job["id"]})


@app.route("/api/2048/solve", methods=["POST"])
def game2048_solve():
    payload = _json_payload()
    policy = payload.get("policy", "hybrid")
    seed = _optional_int(payload.get("seed"))
    max_moves = int(payload.get("max_moves", 2000))
    try:
        agent = _get_game_agent_for_policy(policy, seed)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400

    if payload.get("live"):
        job = _create_job(
            domain="2048",
            kind="solve",
            total=max_moves,
            message="Queued 2048 play",
            metadata={"policy": policy, "max_moves": max_moves},
        )

        def solve_worker() -> None:
            try:
                _patch_job(job["id"], status="running", message=f"Playing 2048 with {policy}")
                steps = agent.play_episode(
                    policy=policy,
                    seed=int(seed) if seed is not None else None,
                    max_moves=max_moves,
                    progress_callback=lambda step: _append_job_step(job["id"], step),
                )
                _finish_job(
                    job["id"],
                    result={
                        "total_steps": len(steps),
                        "best_training_score": agent.best_score,
                        "training_history": _history_with_rolling(agent.training_history),
                        "policy": policy,
                    },
                    message="2048 episode complete",
                )
            except Exception as exc:
                _fail_job(job["id"], exc)

        threading.Thread(target=solve_worker, daemon=True).start()
        return jsonify({"status": "solve started", "job_id": job["id"], "policy": policy})

    steps = agent.play_episode(
        policy=policy,
        seed=int(seed) if seed is not None else None,
        max_moves=max_moves,
    )
    return jsonify({
        "steps": steps,
        "total_steps": len(steps),
        "best_training_score": agent.best_score,
        "training_history": _history_with_rolling(agent.training_history),
        "policy": policy,
    })


@app.route("/api/2048/strategy", methods=["POST"])
def game2048_strategy():
    payload = _json_payload()
    board = payload.get("board")
    if not board or len(board) != 4 or any(len(row) != 4 for row in board):
        return jsonify({"error": "A 4x4 board is required"}), 400

    policy = payload.get("policy", "hybrid")
    seed = _optional_int(payload.get("seed"))
    try:
        agent = _get_game_agent_for_policy(policy, seed)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400

    flat = [int(board[row][col]) for row in range(4) for col in range(4)]
    weighted_cells = []
    for idx, value in enumerate(flat):
        weight = SNAKE_WEIGHTS_2048[idx]
        weighted_cells.append({
            "idx": idx,
            "row": idx // 4,
            "col": idx % 4,
            "tile": value,
            "weight": weight,
            "weighted_value": value * weight,
        })

    return jsonify({
        "policy": policy,
        "snake_weights": SNAKE_WEIGHTS_2048,
        "weighted_cells": weighted_cells,
        "action_values": agent.action_values(board),
    })


@app.route("/api/analysis/compare", methods=["GET", "POST"])
def analysis_compare():
    payload = _json_payload()
    difficulty = payload.get("difficulty") or request.args.get("difficulty", "medium")
    seeds = _parse_comparison_seeds(payload)

    sudoku_agent = _get_or_train_sudoku_for_analysis(difficulty)
    game_agent = _get_or_train_2048_for_analysis()
    comparison = run_comparison(
        sudoku_agent,
        game_agent,
        difficulty=difficulty,
        seeds=seeds,
        game_max_moves=int(payload.get("game_max_moves", 1000)),
    )
    return jsonify(comparison)


@app.route("/api/analysis/export", methods=["POST"])
def analysis_export():
    payload = _json_payload()
    difficulty = payload.get("difficulty", "medium")
    seeds = _parse_comparison_seeds(payload)
    game_max_moves = int(payload.get("game_max_moves", 1000))
    output_dir = payload.get("output_dir")
    if not output_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = ROOT_DIR / "backend" / "artifacts" / "experiments" / f"ui-{timestamp}"

    sudoku_agent = _get_or_train_sudoku_for_analysis(difficulty)
    game_agent = _get_or_train_2048_for_analysis()
    comparison = run_comparison(
        sudoku_agent,
        game_agent,
        difficulty=difficulty,
        seeds=seeds,
        game_max_moves=game_max_moves,
    )
    artifacts = write_experiment_artifacts(
        comparison,
        output_dir,
        sudoku_history=_history_with_rolling(sudoku_agent.training_history),
        game_history=_history_with_rolling(game_agent.training_history),
    )
    return jsonify({
        "comparison": comparison,
        "artifacts": artifacts,
    })


@app.route("/api/analysis/training-curves")
def training_curves():
    sudoku_histories = {
        difficulty: _history_with_rolling(agent.training_history)
        for difficulty, agent in agents["sudoku"].items()
    }
    game_agent = agents["2048"]
    return jsonify({
        "sudoku": sudoku_histories,
        "2048": _history_with_rolling(game_agent.training_history) if game_agent else [],
    })


_load_persisted_agents()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
