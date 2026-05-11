"""
RL Game Solver — Flask Backend
================================
Sudoku  : MRV-guided Q-Learning (feature-state representation)
2048    : Temporal-Difference Q-Learning with feature engineering
          (snake-pattern weight, monotonicity, smoothness, empty cells)
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading

from sudoku.sudoku_rl import SudokuQLearner
from sudoku.sudoku_utils import generate_sudoku

from game2048.game2048_rl import Game2048QLearner

app = Flask(__name__)
CORS(app)

# ─── shared training state ──────────────────────────────────────────────────
training_status = {
    "sudoku": {"easy": "idle", "medium": "idle", "hard": "idle"},
    "2048": "idle"
}
agents = {
    "sudoku": {},   # keyed by difficulty
    "2048": None
}


@app.route("/api/status")
def status():
    return jsonify({"training": {
        "sudoku": training_status["sudoku"],
        "2048": training_status["2048"]
    }})

# ── Sudoku ───────────────────────────────────────────────────────────────────

@app.route("/api/sudoku/train", methods=["POST"])
def sudoku_train():
    diff = request.json.get("difficulty", "medium")
    
    def _train():
        training_status["sudoku"][diff] = "training"
        agent = SudokuQLearner(difficulty=diff)
        eps = {"easy": 200, "medium": 300, "hard": 400}
        agent.train(episodes=eps.get(diff, 300))
        agents["sudoku"][diff] = agent
        training_status["sudoku"][diff] = "ready"
    
    t = threading.Thread(target=_train, daemon=True)
    t.start()
    return jsonify({"status": "training started", "difficulty": diff})

@app.route("/api/sudoku/puzzle")
def sudoku_puzzle():
    diff = request.args.get("difficulty", "medium")
    empty_map = {"easy": 51, "medium": 45, "hard": 51}
    puzzle, _ = generate_sudoku(empty_map.get(diff, 45))
    return jsonify({"board": puzzle, "difficulty": diff})

@app.route("/api/sudoku/solve", methods=["POST"])
def sudoku_solve():
    data = request.json
    diff = data.get("difficulty", "medium")
    agent = agents["sudoku"].get(diff)
    if not agent or not agent.trained:
        return jsonify({"error": "Agent not trained"}), 400

    # Use the board already shown in the UI; fall back to generating one
    puzzle = data.get("board")
    if not puzzle or len(puzzle) != 81:
        empty_map = {"easy": 51, "medium": 45, "hard": 51}
        puzzle, _ = generate_sudoku(empty_map.get(diff, 45))

    steps, solved = agent.solve_with_steps(puzzle)

    return jsonify({
        "puzzle": puzzle,
        "steps": steps,
        "solved": solved,
        "total_steps": len(steps)
    })

# ── 2048 ─────────────────────────────────────────────────────────────────────

@app.route("/api/2048/train", methods=["POST"])
def game2048_train():
    def _train():
        training_status["2048"] = "training"
        agent = Game2048QLearner()
        agent.train(episodes=1000)
        agents["2048"] = agent
        training_status["2048"] = "ready"
    
    t = threading.Thread(target=_train, daemon=True)
    t.start()
    return jsonify({"status": "training started"})

@app.route("/api/2048/solve", methods=["POST"])
def game2048_solve():
    agent = agents["2048"]
    if not agent or not agent.trained:
        return jsonify({"error": "Agent not trained"}), 400
    
    steps = agent.play_episode()
    return jsonify({
        "steps": steps,
        "total_steps": len(steps),
        "best_training_score": agent.best_score
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)