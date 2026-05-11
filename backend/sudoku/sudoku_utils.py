"""
sudoku_utils.py
Utility functions for Sudoku puzzle generation and solving.
"""

import random

def sudoku_empty_count(board, diff):
    counts = {"easy": 51, "medium": 45, "hard": 51}
    return counts.get(diff, 45)

def generate_sudoku(num_empty=45, seed=None):
    """Generate a valid Sudoku puzzle by back-tracking from a full solution."""
    if seed is not None:
        random.seed(seed)
    
    # Build a complete solved board
    board = [0] * 81
    
    def is_valid(b, idx, val):
        r, c = divmod(idx, 9)
        # row
        if val in b[r*9:(r+1)*9]: 
            return False
        # col
        if val in [b[c + i*9] for i in range(9)]: 
            return False
        # box
        br, bc = (r//3)*3, (c//3)*3
        for dr in range(3):
            for dc in range(3):
                if b[(br+dr)*9 + (bc+dc)] == val: 
                    return False
        return True
    
    def fill(pos):
        if pos == 81: 
            return True
        nums = list(range(1, 10))
        random.shuffle(nums)
        for n in nums:
            if is_valid(board, pos, n):
                board[pos] = n
                if fill(pos + 1): 
                    return True
                board[pos] = 0
        return False
    
    fill(0)
    solution = board[:]
    
    # Remove cells
    puzzle = board[:]
    cells = list(range(81))
    random.shuffle(cells)
    removed = 0
    for idx in cells:
        if removed >= num_empty: 
            break
        puzzle[idx] = 0
        removed += 1
    
    return puzzle, solution

def get_candidates(board, idx):
    """Return valid digits for cell idx."""
    if board[idx] != 0:
        return []
    r, c = divmod(idx, 9)
    used = set()
    for i in range(9):
        used.add(board[r*9 + i])
        used.add(board[c + i*9])
    br, bc = (r//3)*3, (c//3)*3
    for dr in range(3):
        for dc in range(3):
            used.add(board[(br+dr)*9 + (bc+dc)])
    return [v for v in range(1, 10) if v not in used]

def mrv_cell(board):
    """Return index of empty cell with Minimum Remaining Values."""
    best_idx, best_count = -1, 10
    for i in range(81):
        if board[i] == 0:
            cands = get_candidates(board, i)
            if not cands: 
                return i, []   # contradiction
            if len(cands) < best_count:
                best_idx, best_count = i, len(cands)
                best_cands = cands
    if best_idx == -1: 
        return -1, []
    return best_idx, best_cands

def state_features(board):
    """
    Compact feature vector → hash key for Q-table.
    Features: (empty_count, conflicts, mrv_min_cands)
    """
    empty = board.count(0)
    # count cells with 1 candidate (naked singles) and 2 candidates
    singles = 0
    doubles = 0
    conflicts = 0
    for i in range(81):
        if board[i] == 0:
            c = get_candidates(board, i)
            if len(c) == 0: 
                conflicts += 1
            elif len(c) == 1: 
                singles += 1
            elif len(c) == 2: 
                doubles += 1
    return (empty // 5, min(singles, 9), min(doubles, 9), min(conflicts, 3))