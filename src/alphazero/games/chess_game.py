"""Chess: AlphaZero-on-chess Game implementation.

Wraps python-chess for rules; uses AlphaZero's 8×8×73 = 4672 action space.
State is a chess.Board (mutable; we copy + push to advance).
"""
from __future__ import annotations

import chess
import numpy as np

from .base import Game
from .chess_move_encoding import ACTION_SIZE, index_to_move, move_to_index

State = chess.Board

NUM_PLANES = 20


class Chess(Game):
    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (NUM_PLANES, 8, 8)

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    def initial_state(self) -> State:
        return chess.Board()

    def current_player(self, state: State) -> int:
        return 1 if state.turn == chess.WHITE else -1

    def legal_actions_mask(self, state: State) -> np.ndarray:
        mask = np.zeros(ACTION_SIZE, dtype=bool)
        for move in state.legal_moves:
            try:
                idx = move_to_index(state, move)
            except ValueError:
                continue  # shouldn't happen for any legal move; safe fallback
            mask[idx] = True
        return mask

    def apply(self, state: State, action: int) -> State:
        move = index_to_move(state, action)
        new_state = state.copy()
        new_state.push(move)
        return new_state

    # Filled in by later tasks
    def terminal_value(self, state: State) -> float | None:
        if not state.is_game_over(claim_draw=True):
            return None
        outcome = state.outcome(claim_draw=True)
        if outcome is None:
            # is_game_over said True but no outcome: edge case, treat as draw
            return 0.0
        if outcome.winner is None:
            return 0.0  # draw (stalemate, 3-fold rep, 50-move, insufficient material)
        # winner is chess.WHITE or chess.BLACK; compare against current player to move
        winner_is_current = (outcome.winner == state.turn)
        return 1.0 if winner_is_current else -1.0

    def encode(self, state: State) -> np.ndarray:
        """Encode the board as a 20-plane (channels, 8, 8) float32 tensor.

        Planes:
            0-5:   my pieces (pawn, knight, bishop, rook, queen, king)
            6-11:  opp pieces (same order)
            12:    ones plane (constant 1)
            13-16: castling rights (my-K, my-Q, opp-K, opp-Q)
            17:    en-passant target square (1-hot at the en-passant square, if any)
            18:    halfmove clock for 50-move rule (normalized: clock / 100)
            19:    fullmove number normalized: fullmove_number / 100
        """
        enc = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
        my_color = state.turn  # current player
        opp_color = not my_color
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                       chess.ROOK, chess.QUEEN, chess.KING]

        # Planes 0-5: my pieces
        for plane_idx, pt in enumerate(piece_types):
            for sq in state.pieces(pt, my_color):
                r, f = chess.square_rank(sq), chess.square_file(sq)
                enc[plane_idx, r, f] = 1.0
        # Planes 6-11: opp pieces
        for plane_idx, pt in enumerate(piece_types):
            for sq in state.pieces(pt, opp_color):
                r, f = chess.square_rank(sq), chess.square_file(sq)
                enc[6 + plane_idx, r, f] = 1.0
        # Plane 12: ones
        enc[12, :, :] = 1.0
        # Planes 13-16: castling rights
        if state.has_kingside_castling_rights(my_color):
            enc[13, :, :] = 1.0
        if state.has_queenside_castling_rights(my_color):
            enc[14, :, :] = 1.0
        if state.has_kingside_castling_rights(opp_color):
            enc[15, :, :] = 1.0
        if state.has_queenside_castling_rights(opp_color):
            enc[16, :, :] = 1.0
        # Plane 17: en passant target
        if state.ep_square is not None:
            r, f = chess.square_rank(state.ep_square), chess.square_file(state.ep_square)
            enc[17, r, f] = 1.0
        # Plane 18: halfmove clock normalized
        enc[18, :, :] = min(state.halfmove_clock / 100.0, 1.0)
        # Plane 19: fullmove number normalized
        enc[19, :, :] = min(state.fullmove_number / 100.0, 1.0)
        return enc

    def canonical_state(self, state: State) -> State:
        """Return state from the current player's POV.

        For chess: when Black is to move, mirror the board so the network
        always sees the to-move player as "White at the bottom." python-chess's
        Board.mirror() handles castling rights and en passant correctly.
        """
        if state.turn == chess.BLACK:
            return state.mirror()
        return state

    def symmetries(self, encoded, policy):
        """Chess has no symmetries (castling rights and pawn direction
        break left-right mirror; player asymmetry breaks rotation)."""
        return [(encoded, policy)]
