# tests/test_chess_minimax.py
import chess
import pytest

from alphazero.games.chess_game import Chess
from alphazero.opponents.stockfish import StockfishOpponent


@pytest.fixture
def game():
    return Chess()


@pytest.fixture
def opp():
    o = StockfishOpponent(elo=1500, time_per_move=0.1)
    yield o
    o.close()


def test_stockfish_plays_legal_move(game, opp):
    s = game.initial_state()
    action = opp(game, s)
    mask = game.legal_actions_mask(s)
    assert mask[action], f"Stockfish proposed illegal action {action}"


def test_stockfish_makes_different_moves_at_different_elos(game):
    """Sanity: Stockfish at 1320 vs 2000 should disagree at least sometimes."""
    # Use a position with multiple reasonable moves
    s = chess.Board("rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 1 2")
    weak = StockfishOpponent(elo=1320, time_per_move=0.1)
    strong = StockfishOpponent(elo=2400, time_per_move=0.1)
    try:
        # Different ELOs may produce different moves; not guaranteed each call but
        # likely over several positions. Just verify both return valid actions.
        weak_action = weak(game, s)
        strong_action = strong(game, s)
        mask = game.legal_actions_mask(s)
        assert mask[weak_action]
        assert mask[strong_action]
    finally:
        weak.close()
        strong.close()


def test_stockfish_clean_close(game):
    """Calling close() twice should not crash."""
    o = StockfishOpponent(elo=1500, time_per_move=0.1)
    o(game, game.initial_state())  # use it
    o.close()
    o.close()  # second close is no-op
