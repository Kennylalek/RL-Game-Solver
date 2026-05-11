"""
sudoku_rl.py
Sudoku Q-Learning Agent Class
Q-Learning with MRV (Minimum Remaining Values) heuristic
"""

from collections import defaultdict
import random
from sudoku.sudoku_utils import (
    generate_sudoku, get_candidates, mrv_cell, state_features
)

class SudokuQLearner:
    def __init__(self, difficulty="medium"):
        self.diff = difficulty
        empty_map = {"easy": 51, "medium": 45, "hard": 51}
        self.num_empty = empty_map[difficulty]
        self.q = defaultdict(float)
        self.alpha = 0.3
        self.gamma = 0.95
        self.epsilon = 1.0
        self.eps_decay = 0.995
        self.eps_min = 0.05
        self.trained = False

    def choose_action(self, board):
        """Use MRV + epsilon-greedy."""
        idx, cands = mrv_cell(board)
        if idx == -1 or not cands:
            return None
        
        state = state_features(board)
        
        if random.random() < self.epsilon:
            return (idx, random.choice(cands))
        
        # Greedy over candidates for MRV cell
        best_val, best_q = cands[0], -1e9
        for v in cands:
            key = (state, idx, v)
            q = self.q[key]
            if q > best_q:
                best_q, best_val = q, v
        return (idx, best_val)

    def update(self, state, action, reward, next_state, done):
        if action is None: 
            return
        idx, val = action
        key = (state, idx, val)
        if done:
            target = reward
        else:
            # max Q for next state  (approximation over same cell candidates)
            target = reward + self.gamma * 0.5  # bootstrap
        self.q[key] += self.alpha * (target - self.q[key])

    def train(self, episodes=300):
        for ep in range(episodes):
            puzzle, _ = generate_sudoku(self.num_empty, seed=random.randint(0, 9999))
            board = puzzle[:]
            
            for step in range(200):
                state = state_features(board)
                action = self.choose_action(board)
                if action is None: 
                    break
                
                idx, val = action
                board[idx] = val
                
                # Reward shaping
                empty = board.count(0)
                if empty == 0:
                    reward = 10.0
                    done = True
                else:
                    # Penalise contradictions
                    has_conflict = any(
                        len(get_candidates(board, j)) == 0
                        for j in range(81) if board[j] == 0
                    )
                    reward = -5.0 if has_conflict else 0.2
                    done = has_conflict
                
                next_state = state_features(board)
                self.update(state, action, reward, next_state, done)
                
                if done and reward < 0:
                    # backtrack: undo last placement
                    board[idx] = 0
                    break
                if board.count(0) == 0:
                    break
            
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        
        self.trained = True

    def solve_with_steps(self, puzzle):
        """
        Produce step-by-step trace using trained policy + MRV backtracking.
        """
        board = puzzle[:]
        steps = [{"board": board[:], "action": None,
                   "message": "Puzzle loaded", "reward": 0}]
        
        stack = []  # for backtracking: list of (board_snapshot, cell_idx, tried_vals)
        
        for _ in range(5000):
            idx, cands = mrv_cell(board)
            
            if idx == -1:
                # Solved!
                steps.append({"board": board[:], "action": None,
                               "message": "✅ SOLVED!", "reward": 10})
                return steps, True
            
            if not cands:
                # Contradiction — backtrack
                if not stack:
                    break
                board, sidx, remaining = stack.pop()
                if not remaining:
                    continue
                val = remaining.pop()
                board[sidx] = val
                steps.append({"board": board[:], "action": {"idx": sidx, "val": val},
                               "message": f"↩ Backtrack → cell {sidx}='{val}'",
                               "reward": -1})
                stack.append((board[:], sidx, remaining))
                continue
            
            # Q-guided choice
            state = state_features(board)
            best_val = cands[0]
            best_q = -1e9
            for v in cands:
                q = self.q.get((state, idx, v), 0.0)
                if q > best_q:
                    best_q, best_val = q, v
            
            remaining = [v for v in cands if v != best_val]
            stack.append((board[:], idx, remaining))
            board[idx] = best_val
            
            steps.append({
                "board": board[:],
                "action": {"idx": idx, "val": best_val},
                "message": f"Place {best_val} at ({idx//9},{idx%9}) | Q={best_q:.2f}",
                "reward": round(best_q, 2)
            })
        
        # Return partial solution if unsolved
        steps.append({"board": board[:], "action": None,
                       "message": "⚠ Partial solution (increase training)", "reward": 0})
        return steps, board.count(0) == 0