# RL Game Solver

Sudoku and 2048 reinforcement-learning project with a Flask backend and a static browser UI.

## Implemented Methods

### Sudoku

- **RL Q + MRV**: tabular Q-learning over compact state features. MRV selects the constrained cell; Q-values rank candidate digits.
- **MRV Heuristic**: deterministic backtracking using the minimum-remaining-values rule.
- **Pure Backtracking**: deterministic first-empty-cell backtracking.
- **Random MRV**: MRV cell selection with randomized candidate ordering.

The UI exposes these methods in the Sudoku **Algorithm** selector. Baseline methods can solve without training; `RL Q + MRV` requires the matching difficulty agent to be trained.

Sudoku state features are:

```text
(empty bucket, naked singles, double-candidate cells, contradictions)
```

Generated public puzzles are checked for exactly one solution.

### 2048

- **RL Q Policy**: greedy policy from the learned Q-table.
- **Expectimax**: depth-2 stochastic lookahead using tile-spawn probabilities.
- **Hybrid Q + Expectimax**: expectimax with learned Q-values added at leaf evaluation.
- **Random Policy**: valid random moves.

2048 state features are:

```text
(max log2 tile, empty-cell bucket, monotonicity, smoothness bucket, snake-pattern bucket)
```

## Setup

```bash
cd /Users/elyashadjar/Dev/RL-Game-Solver
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

Terminal 1:

```bash
cd /Users/elyashadjar/Dev/RL-Game-Solver/backend
../.venv/bin/python app.py
```

Terminal 2:

```bash
cd /Users/elyashadjar/Dev/RL-Game-Solver/frontend
../.venv/bin/python -m http.server 5173
```

Open:

```text
http://127.0.0.1:5173/index.html
```

## Live Jobs

Training and solving can run as live background jobs. The start endpoint returns a `job_id`, then the UI polls the job endpoint for incremental history or step data.

```bash
curl -X POST http://127.0.0.1:5050/api/sudoku/train \
  -H 'Content-Type: application/json' \
  -d '{"difficulty":"medium","episodes":300}'

curl http://127.0.0.1:5050/api/jobs/<job_id>
```

Live solve mode:

```bash
curl -X POST http://127.0.0.1:5050/api/sudoku/solve \
  -H 'Content-Type: application/json' \
  -d '{"difficulty":"medium","algorithm":"mrv","board":[...],"live":true}'

curl -X POST http://127.0.0.1:5050/api/2048/solve \
  -H 'Content-Type: application/json' \
  -d '{"policy":"hybrid","live":true,"max_moves":1000}'
```

The comparison board is run from the **Comparison controls** section in the UI. Set Sudoku difficulty, seed, run count, and 2048 move cap, then click **Run comparison**. The same section can export CSV, JSON, and PNG artifacts for the report.

## Test

```bash
cd /Users/elyashadjar/Dev/RL-Game-Solver
.venv/bin/python -m pytest
```

## Experiments

```bash
cd /Users/elyashadjar/Dev/RL-Game-Solver
.venv/bin/python scripts/run_experiments.py \
  --difficulty medium \
  --seeds 101,202,303,404,505 \
  --sudoku-episodes 200 \
  --game-episodes 500 \
  --game-max-moves 1000 \
  --output-dir artifacts/experiments
```

Outputs:

- `artifacts/experiments/sudoku_comparison.csv`
- `artifacts/experiments/game2048_comparison.csv`
- `artifacts/experiments/summary.json`
- `artifacts/experiments/sudoku_training_curve.png`
- `artifacts/experiments/game2048_training_curve.png`

## Standout Features

1. Policy replay with step explanations for Sudoku candidates and 2048 action values.
2. Side-by-side replay for two selected methods on the same seed or Sudoku board.
3. Results dashboard that summarizes the strongest Sudoku and 2048 methods after comparison.
4. UI-controlled Q-learning hyperparameters: learning rate, discount, exploration, decay, and minimum exploration.
5. Save, load, and reset controls for persisted trained Q-table models.
6. Seeded reproducibility for puzzles, 2048 runs, training, comparisons, and replay.
7. 2048 strategy heatmap using the snake-pattern tile priority used by the heuristic.
8. Sudoku difficulty validation with empty-cell count, uniqueness, MRV effort, and backtracking effort.
9. Rolling training metrics for smoother reward, filled-cell, score, and max-tile curves.
10. Report artifact export from the UI: Sudoku CSV, 2048 CSV, summary JSON, and training plots.
