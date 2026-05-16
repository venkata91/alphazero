"""StockfishOpponent — UCI wrapper for the Stockfish engine, used as the
chess eval baseline at a configured ELO.

Requires 'stockfish' binary on PATH. See bin/setup.sh for install instructions.
"""
from __future__ import annotations

import chess
import chess.engine

from ..games.chess_game import Chess
from ..games.chess_move_encoding import move_to_index


class StockfishOpponent:
    def __init__(
        self,
        elo: int = 1500,
        time_per_move: float = 0.5,
        stockfish_path: str = "stockfish",
    ):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self._closed = False  # set BEFORE configure so close() works if configure raises
        try:
            self.engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": elo,
            })
        except Exception:
            self.close()
            raise
        self.time_per_move = time_per_move

    def __call__(self, game: Chess, state: chess.Board) -> int:
        result = self.engine.play(state, chess.engine.Limit(time=self.time_per_move))
        return move_to_index(state, result.move)

    def reset(self) -> None:
        # Stockfish handles fresh games internally; no per-game setup needed.
        pass

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.engine.quit()
        except (chess.engine.EngineTerminatedError, BrokenPipeError):
            pass
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
