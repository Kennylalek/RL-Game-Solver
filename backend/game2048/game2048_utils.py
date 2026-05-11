"""
game2048_utils.py
Utility functions for 2048 game state management and feature extraction.
"""

import random
import math

ACTIONS_2048 = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

def new_board_2048():
    b = [[0]*4 for _ in range(4)]
    add_tile(b)
    add_tile(b)
    return b

def add_tile(board):
    empties = [(r,c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if empties:
        r, c = random.choice(empties)
        board[r][c] = 4 if random.random() < 0.1 else 2

def slide_row_left(row):
    tiles = [x for x in row if x != 0]
    merged = []
    score = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i+1]:
            val = tiles[i] * 2
            merged.append(val)
            score += val
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    return merged + [0]*(4-len(merged)), score

def move_2048(board, action):
    """Returns (new_board, score_gain, moved)."""
    b = [row[:] for row in board]
    total_score = 0
    moved = False
    
    if action == 2:   # LEFT
        for r in range(4):
            new_row, s = slide_row_left(b[r])
            if new_row != b[r]: 
                moved = True
            b[r] = new_row
            total_score += s
    elif action == 3: # RIGHT
        for r in range(4):
            rev, s = slide_row_left(b[r][::-1])
            new_row = rev[::-1]
            if new_row != b[r]: 
                moved = True
            b[r] = new_row
            total_score += s
    elif action == 0: # UP
        for c in range(4):
            col = [b[r][c] for r in range(4)]
            new_col, s = slide_row_left(col)
            for r in range(4):
                if b[r][c] != new_col[r]: 
                    moved = True
                b[r][c] = new_col[r]
            total_score += s
    elif action == 1: # DOWN
        for c in range(4):
            col = [b[r][c] for r in range(4)][::-1]
            new_col, s = slide_row_left(col)
            new_col = new_col[::-1]
            for r in range(4):
                if b[r][c] != new_col[r]: 
                    moved = True
                b[r][c] = new_col[r]
            total_score += s
    
    return b, total_score, moved

def is_game_over_2048(board):
    for r in range(4):
        for c in range(4):
            if board[r][c] == 0:
                return False
            if c < 3 and board[r][c] == board[r][c+1]: 
                return False
            if r < 3 and board[r][c] == board[r+1][c]: 
                return False
    return True

def board_features(board):
    """
    Feature engineering for 2048 Q-table key:
    - max tile (log2 bucket)
    - empty cells (bucketed)
    - monotonicity score bucket
    - smoothness bucket
    - snake pattern score bucket
    """
    flat = [board[r][c] for r in range(4) for c in range(4)]
    max_tile = max(flat)
    max_log = int(math.log2(max_tile)) if max_tile > 0 else 0
    empty = flat.count(0)
    
    # Monotonicity (prefer rows/cols to be monotone)
    mono = 0
    for r in range(4):
        row = board[r]
        if all(row[i] >= row[i+1] for i in range(3)) or \
           all(row[i] <= row[i+1] for i in range(3)):
            mono += 1
    for c in range(4):
        col = [board[r][c] for r in range(4)]
        if all(col[i] >= col[i+1] for i in range(3)) or \
           all(col[i] <= col[i+1] for i in range(3)):
            mono += 1
    
    # Snake weight (high values in snake pattern)
    snake_weights = [
        15,14,13,12,
         8, 9,10,11,
         7, 6, 5, 4,
         0, 1, 2, 3
    ]
    snake_score = sum(flat[i] * snake_weights[i] for i in range(16))
    snake_bucket = min(int(math.log2(snake_score + 1)) // 2, 10) if snake_score > 0 else 0
    
    return (max_log, min(empty, 8), min(mono, 8), snake_bucket)