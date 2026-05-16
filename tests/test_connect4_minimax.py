import numpy as np
import pytest

from alphazero.games.connect4 import Connect4
from alphazero.opponents.connect4_minimax import Connect4MinimaxOpponent


@pytest.fixture
def c4() -> Connect4:
    return Connect4()


def _board(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def test_only_plays_legal_moves(c4: Connect4):
    """Minimax must never propose an illegal action."""
    opp = Connect4MinimaxOpponent(depth=4)
    s = c4.initial_state()
    for _ in range(6):
        s = c4.apply(s, 0)  # fill column 0
    action = opp(c4, s)
    legal = c4.legal_actions_mask(s)
    assert legal[action], f"Minimax returned illegal action {action}"


def test_takes_immediate_winning_move(c4: Connect4):
    """+1 can win at column 3. Setup: 3X bottom row + 3O elsewhere = 6 (+1 to move)."""
    state = _board([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, -1, 0, 0],
    ])
    assert c4.current_player(state) == 1
    opp = Connect4MinimaxOpponent(depth=4)
    action = opp(c4, state)
    assert action == 3, f"Should play winning move (col 3), got {action}"


def test_blocks_immediate_opponent_threat(c4: Connect4):
    """-1 has 3 in a row. +1 to move MUST block at col 3."""
    state = _board([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, 0, 1, 0, 0],
    ])
    assert c4.current_player(state) == 1
    opp = Connect4MinimaxOpponent(depth=4)
    action = opp(c4, state)
    assert action == 3, f"Must block at col 3, got {action}"


def test_prefers_center_on_empty_board(c4: Connect4):
    """Minimax with center-bias heuristic should pick column 3 on the first move."""
    opp = Connect4MinimaxOpponent(depth=4)
    action = opp(c4, c4.initial_state())
    assert action == 3, f"Expected center (col 3) opening, got {action}"


def test_depth_8_beats_depth_2(c4: Connect4):
    """Deeper search should produce strictly stronger play."""
    from alphazero.arena import play_match
    strong = Connect4MinimaxOpponent(depth=8)
    weak = Connect4MinimaxOpponent(depth=2)
    np.random.seed(0)
    result = play_match(c4, strong, weak, num_games=10)
    assert result.win_rate >= 0.70, (
        f"depth-8 only achieved win_rate {result.win_rate:.2%} vs depth-2; "
        f"expected ≥70% (with deterministic seeded play, depth-8 wins as +1 "
        f"and draws as -1 → ~75% with the current heuristic)"
    )
