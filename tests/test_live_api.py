from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import app as app_module
from app import app
from game2048.game2048_rl import Game2048QLearner
from sudoku.sudoku_rl import SudokuQLearner


def test_live_training_job_endpoint_exposes_progress() -> None:
    client = app.test_client()
    started = client.post("/api/sudoku/train", json={"difficulty": "easy", "episodes": 1})
    assert started.status_code == 200
    job_id = started.get_json()["job_id"]

    job = None
    for _ in range(40):
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.get_json()
        if job["status"] == "complete":
            break
        time.sleep(0.05)

    assert job is not None
    assert job["status"] == "complete"
    assert job["progress"] == 1
    assert len(job["history"]) == 1
    assert "rolling_reward" in job["history"][0]


def test_sudoku_baseline_solve_does_not_require_rl_agent() -> None:
    client = app.test_client()
    puzzle = client.get("/api/sudoku/puzzle?difficulty=easy").get_json()["board"]

    response = client.post(
        "/api/sudoku/solve",
        json={"difficulty": "hard", "algorithm": "mrv", "board": puzzle},
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["algorithm"] == "mrv"
    assert payload["solved"] is True
    assert payload["training_history"] == []
    assert payload["validation"]["unique_solution"] is True


def test_2048_expectimax_solve_does_not_require_trained_agent() -> None:
    client = app.test_client()
    previous_agent = app_module.agents["2048"]
    app_module.agents["2048"] = None

    try:
        response = client.post(
            "/api/2048/solve",
            json={"policy": "expectimax", "seed": 7, "max_moves": 5},
        )
    finally:
        app_module.agents["2048"] = previous_agent

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["policy"] == "expectimax"
    assert payload["total_steps"] >= 1


def test_2048_strategy_endpoint_returns_heatmap_weights() -> None:
    client = app.test_client()
    response = client.post(
        "/api/2048/strategy",
        json={
            "policy": "expectimax",
            "board": [
                [2, 4, 8, 16],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert len(payload["weighted_cells"]) == 16
    assert payload["snake_weights"][0] == 15


def test_sudoku_model_actions_use_persistence(monkeypatch, tmp_path: Path) -> None:
    client = app.test_client()
    monkeypatch.setattr(app_module, "SUDOKU_AGENT_DIR", tmp_path)
    agent = SudokuQLearner("easy", seed=3)
    agent.train(episodes=1)
    app_module.agents["sudoku"]["easy"] = agent
    app_module.training_status["sudoku"]["easy"] = "ready"

    saved = client.post("/api/sudoku/model", json={"difficulty": "easy", "action": "save"})
    assert saved.status_code == 200
    assert Path(saved.get_json()["path"]).exists()

    app_module.agents["sudoku"].pop("easy", None)
    app_module.training_status["sudoku"]["easy"] = "idle"
    loaded = client.post("/api/sudoku/model", json={"difficulty": "easy", "action": "load"})
    assert loaded.status_code == 200
    assert loaded.get_json()["q_size"] >= 0

    reset = client.post("/api/sudoku/model", json={"difficulty": "easy", "action": "reset"})
    assert reset.status_code == 200
    assert not Path(reset.get_json()["path"]).exists()


def test_export_artifacts_endpoint_writes_report_files(tmp_path: Path) -> None:
    client = app.test_client()
    sudoku_agent = SudokuQLearner("easy", seed=5)
    sudoku_agent.train(episodes=1)
    game_agent = Game2048QLearner(seed=6)
    game_agent.train(episodes=1)
    app_module.agents["sudoku"]["easy"] = sudoku_agent
    app_module.agents["2048"] = game_agent
    app_module.training_status["sudoku"]["easy"] = "ready"
    app_module.training_status["2048"] = "ready"

    response = client.post(
        "/api/analysis/export",
        json={
            "difficulty": "easy",
            "runs": 1,
            "seed": 101,
            "game_max_moves": 5,
            "output_dir": str(tmp_path / "experiment"),
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert Path(payload["artifacts"]["sudoku_csv"]).exists()
    assert Path(payload["artifacts"]["game2048_csv"]).exists()
    assert Path(payload["artifacts"]["summary_json"]).exists()
