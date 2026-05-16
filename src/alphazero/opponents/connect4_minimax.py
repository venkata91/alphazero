"""Connect 4 α-β minimax opponent with heuristic position evaluation.

Used as the eval ground truth for Sub-project 2 (we don't have a perfect
solver for Connect 4 from scratch — depth-8 minimax is a strong amateur).

Heuristic eval at non-terminal leaves:
    - Score sliding 4-cell windows in all 4 directions
        4-of-mine        → +1000 (effectively a forced win)
        3-of-mine + 1 empty → +5
        2-of-mine + 2 empties → +2
        symmetric negatives for opponent
        opp 3-in-a-row gets -50 (block aggressively even before center bias)
    - +3 per piece in the center column (mine); -3 per opponent center piece

Performance:
    - Fully vectorised window scoring (no Python loop over windows).
    - Center-first move ordering maximises α-β pruning.
    - Transposition table (Zobrist hashing) avoids revisiting positions.
    - In-place board modification avoids ndarray allocation per node.
    - Incremental win detection only checks lines through the last move.
"""
from __future__ import annotations

import numpy as np

from ..games.base import Game, State
from ..games.connect4 import COLS, ROWS, Connect4

CENTER_COL = COLS // 2  # column 3 (the center of a 7-col board)
WIN_SCORE = 1_000.0
BLOCK_SCORE = -40.0

# Move ordering: center-first maximises α-β cutoffs.
# For COLS=7: [3, 2, 4, 1, 5, 0, 6]
_MOVE_ORDER: list[int] = sorted(range(COLS), key=lambda c: abs(c - CENTER_COL))

# Zobrist hash table: shape (ROWS, COLS, 3) — indices 0=empty,1=+1,2=-1
_rng = np.random.default_rng(0)
_ZOBRIST = _rng.integers(0, 2**63, size=(ROWS, COLS, 3), dtype=np.int64)

# Transposition table entry flags
_EXACT = 0
_LOWER = 1  # value is a lower bound (alpha-cutoff)
_UPPER = 2  # value is an upper bound (beta-cutoff)

# ---------------------------------------------------------------------------
# Pre-computed window index arrays for vectorised evaluation
# ---------------------------------------------------------------------------
def _build_window_indices() -> list[np.ndarray]:
    """Return a list of (4, 2) arrays, each row being (row, col) of a cell."""
    windows = []
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            windows.append(np.array([(r, c + i) for i in range(4)]))
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            windows.append(np.array([(r + i, c) for i in range(4)]))
    # Diagonal down-right
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            windows.append(np.array([(r + i, c + i) for i in range(4)]))
    # Anti-diagonal
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            windows.append(np.array([(r - i, c + i) for i in range(4)]))
    return windows


_WIN_IDX = _build_window_indices()
_N_WINDOWS = len(_WIN_IDX)

# Stack all window indices into a single (N_WINDOWS, 4, 2) array
_WIN_ROWS = np.array([[idx[i][0] for i in range(4)] for idx in _WIN_IDX])  # (N, 4)
_WIN_COLS = np.array([[idx[i][1] for i in range(4)] for idx in _WIN_IDX])  # (N, 4)


def _fast_evaluate_vec(board: np.ndarray, player: int) -> float:
    """Fully vectorised heuristic — no Python loops over windows."""
    # Extract all windows at once: shape (N_WINDOWS, 4)
    w = board[_WIN_ROWS, _WIN_COLS]  # (N, 4), values in {-1, 0, +1}

    my  = (w == player).sum(axis=1)   # (N,)
    opp = (w == -player).sum(axis=1)  # (N,)

    # Mixed windows contribute 0
    pure = (my == 0) | (opp == 0)

    score = np.zeros(_N_WINDOWS, dtype=np.float64)

    # My windows — aggressive offensive scoring
    score += np.where(pure & (my == 4),  WIN_SCORE,  0.0)
    score += np.where(pure & (my == 3),  50.0,       0.0)
    score += np.where(pure & (my == 2),  5.0,        0.0)

    # Opponent windows
    score += np.where(pure & (opp == 4), -WIN_SCORE, 0.0)
    score += np.where(pure & (opp == 3), BLOCK_SCORE, 0.0)
    score += np.where(pure & (opp == 2), -5.0,        0.0)

    total = float(score.sum())

    # Center column bonus
    center = board[:, CENTER_COL]
    total += 3.0 * int((center == player).sum())
    total -= 3.0 * int((center == -player).sum())
    return total


def _piece_index(v: int) -> int:
    return 0 if v == 0 else (1 if v == 1 else 2)


def _board_hash(board: np.ndarray) -> int:
    h = np.int64(0)
    for r in range(ROWS):
        for c in range(COLS):
            h ^= _ZOBRIST[r, c, _piece_index(int(board[r, c]))]
    return int(h)


def _drop_row(board: np.ndarray, col: int) -> int:
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == 0:
            return r
    return -1


def _check_win(board: np.ndarray, player: int, row: int, col: int) -> bool:
    """Check only lines through (row, col) — O(4*7) instead of full scan."""
    p = np.int8(player)

    def _run(dr: int, dc: int) -> int:
        cnt = 0
        r, c = row + dr, col + dc
        while 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == p:
            cnt += 1
            r += dr
            c += dc
        return cnt

    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if 1 + _run(dr, dc) + _run(-dr, -dc) >= 4:
            return True
    return False


def _is_draw(board: np.ndarray) -> bool:
    return bool((board[0] != 0).all())


def _current_player(board: np.ndarray) -> int:
    pieces = int((board != 0).sum())
    return 1 if pieces % 2 == 0 else -1


class Connect4MinimaxOpponent:
    """α-β minimax to a fixed depth with center-bias heuristic + transposition table."""

    def __init__(self, depth: int = 8):
        self.depth = depth
        self._game = Connect4()
        self._tt: dict[int, tuple[int, int, float]] = {}

    def reset(self) -> None:
        """Clear transposition table for a new game."""
        self._tt.clear()

    def __call__(self, game: Game, state: State) -> int:
        """Return the best action for current_player at this state."""
        legal = game.legal_actions_mask(state)
        legal_cols = [c for c in _MOVE_ORDER if legal[c]]
        if not legal_cols:
            raise ValueError("No legal actions at this state")

        board = state.copy()
        h = _board_hash(board)
        player = _current_player(board)

        best_action = legal_cols[0]
        best_value = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        for col in legal_cols:
            row = _drop_row(board, col)
            board[row, col] = np.int8(player)
            new_h = h ^ _ZOBRIST[row, col, 0] ^ _ZOBRIST[row, col, _piece_index(player)]
            if _check_win(board, player, row, col):
                board[row, col] = 0
                return col  # immediate win, no search needed
            value = -self._negamax(board, new_h, self.depth - 1, -beta, -alpha, -player)
            board[row, col] = 0
            if value > best_value:
                best_value = value
                best_action = col
            alpha = max(alpha, best_value)
        return best_action

    def _negamax(
        self,
        board: np.ndarray,
        h: int,
        depth: int,
        alpha: float,
        beta: float,
        player: int,
    ) -> float:
        orig_alpha = alpha

        # Transposition table lookup
        tt_entry = self._tt.get(h)
        if tt_entry is not None:
            tt_depth, tt_flag, tt_val = tt_entry
            if tt_depth >= depth:
                if tt_flag == _EXACT:
                    return tt_val
                if tt_flag == _LOWER:
                    alpha = max(alpha, tt_val)
                elif tt_flag == _UPPER:
                    beta = min(beta, tt_val)
                if alpha >= beta:
                    return tt_val

        if _is_draw(board):
            return 0.0
        if depth == 0:
            return _fast_evaluate_vec(board, player)

        best_value = -float("inf")
        for col in _MOVE_ORDER:
            row = _drop_row(board, col)
            if row == -1:
                continue
            board[row, col] = np.int8(player)
            new_h = h ^ _ZOBRIST[row, col, 0] ^ _ZOBRIST[row, col, _piece_index(player)]
            if _check_win(board, player, row, col):
                board[row, col] = 0
                best_value = WIN_SCORE
                break  # can't do better — prune immediately
            value = -self._negamax(board, new_h, depth - 1, -beta, -alpha, -player)
            board[row, col] = 0
            best_value = max(best_value, value)
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break  # β-cutoff

        # Update transposition table
        if best_value <= orig_alpha:
            flag = _UPPER
        elif best_value >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        self._tt[h] = (depth, flag, best_value)

        return best_value
