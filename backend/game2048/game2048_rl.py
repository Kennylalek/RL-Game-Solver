"""
game2048_rl.py
RL agent for 2048 game, using Q-learning with reward shaping and Expectimax-1 lookahead during play.
"""

import math
import random
from collections import defaultdict
from game2048.game2048_utils import new_board_2048, move_2048, add_tile, is_game_over_2048, board_features, ACTIONS_2048, ACTION_NAMES

class Game2048QLearner:
    def __init__(self):
        self.q = defaultdict(float)
        self.alpha = 0.2
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_decay = 0.997
        self.eps_min = 0.05
        self.trained = False
        self.best_score = 0

    def choose_action(self, board):
        """Epsilon-greedy with valid-move masking."""
        valid = [a for a in ACTIONS_2048
                 if move_2048(board, a)[2]]
        if not valid:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid)
        
        state = board_features(board)
        best_a, best_q = valid[0], -1e9
        for a in valid:
            q = self.q[(state, a)]
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    def update(self, state, action, reward, next_board, done):
        if action is None: 
            return
        key = (state, action)
        if done:
            target = reward
        else:
            ns = board_features(next_board)
            future = max(self.q[(ns, a)] for a in ACTIONS_2048)
            target = reward + self.gamma * future
        self.q[key] += self.alpha * (target - self.q[key])

    def reward_shaping(self, board, new_board, score_gain):
        """Shaped reward: merge score + empty cells bonus + monotonicity."""
        flat_new = [new_board[r][c] for r in range(4) for c in range(4)]
        empty_bonus = flat_new.count(0) * 2
        max_tile = max(flat_new)
        # Reward log2 of max tile achieved
        max_bonus = math.log2(max_tile) if max_tile > 0 else 0
        return math.log2(score_gain + 1) * 10 + empty_bonus + max_bonus * 0.5

    def train(self, episodes=1000):
        for ep in range(episodes):
            board = new_board_2048()
            episode_score = 0
            
            for _ in range(2000):
                state = board_features(board)
                action = self.choose_action(board)
                if action is None: 
                    break
                
                new_board, score_gain, moved = move_2048(board, action)
                if not moved:
                    self.update(state, action, -5, board, False)
                    continue
                
                episode_score += score_gain
                add_tile(new_board)
                done = is_game_over_2048(new_board)
                
                reward = self.reward_shaping(board, new_board, score_gain)
                if done: 
                    reward -= 20
                
                self.update(state, action, reward, new_board, done)
                board = new_board
                if done: 
                    break
            
            self.best_score = max(self.best_score, episode_score)
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        
        self.trained = True

    def play_episode(self):
        """Run one greedy episode, capturing all steps."""
        board = new_board_2048()
        steps = []
        total_score = 0
        
        steps.append({
            "board": [row[:] for row in board],
            "action": None,
            "action_name": None,
            "score": 0,
            "reward": 0,
            "message": "Game started"
        })
        
        old_eps = self.epsilon
        self.epsilon = 0.05  # nearly greedy during play
        
        for move_num in range(2000):
            # Use Expectimax-1 lookahead for play (not training)
            action = self.expectimax_action(board, depth=2)
            if action is None:
                break
            
            new_board, score_gain, moved = move_2048(board, action)
            if not moved:
                action = self.choose_action(board)
                if action is None: 
                    break
                new_board, score_gain, moved = move_2048(board, action)
                if not moved: 
                    break
            
            total_score += score_gain
            add_tile(new_board)
            done = is_game_over_2048(new_board)
            
            reward = self.reward_shaping(board, new_board, score_gain)
            
            steps.append({
                "board": [row[:] for row in new_board],
                "action": action,
                "action_name": ACTION_NAMES[action],
                "score": total_score,
                "reward": round(reward, 2),
                "message": f"Move {move_num+1}: {ACTION_NAMES[action]} | Score: {total_score}"
            })
            
            board = new_board
            if done:
                max_tile = max(board[r][c] for r in range(4) for c in range(4))
                steps[-1]["message"] = f"Game Over | Score: {total_score} | Max tile: {max_tile}"
                break
        
        self.epsilon = old_eps
        return steps

    def expectimax_action(self, board, depth=2):
        """1-ply Expectimax: pick action maximising expected value over tile spawns."""
        valid = [a for a in ACTIONS_2048 if move_2048(board, a)[2]]
        if not valid: 
            return None
        
        best_a, best_v = valid[0], -1e9
        for a in valid:
            new_b, score, _ = move_2048(board, a)
            val = score + self._expect(new_b, depth - 1)
            if val > best_v:
                best_v, best_a = val, a
        return best_a

    def _expect(self, board, depth):
        if depth == 0:
            flat = [board[r][c] for r in range(4) for c in range(4)]
            empty = flat.count(0)
            max_t = max(flat)
            return (math.log2(max_t) if max_t > 0 else 0) * 50 + empty * 10
        
        empties = [(r,c) for r in range(4) for c in range(4) if board[r][c] == 0]
        if not empties:
            return self._max_val(board, depth)
        
        # Sample at most 4 empties for speed
        sample = empties[:4]
        total = 0
        for r, c in sample:
            for val, prob in [(2, 0.9), (4, 0.1)]:
                nb = [row[:] for row in board]
                nb[r][c] = val
                total += prob * self._max_val(nb, depth - 1)
        return total / len(sample)

    def _max_val(self, board, depth):
        valid = [a for a in ACTIONS_2048 if move_2048(board, a)[2]]
        if not valid or depth == 0:
            flat = [board[r][c] for r in range(4) for c in range(4)]
            return (math.log2(max(flat)) if max(flat) > 0 else 0) * 50
        return max(move_2048(board, a)[1] + self._expect(move_2048(board, a)[0], depth-1)
                   for a in valid)