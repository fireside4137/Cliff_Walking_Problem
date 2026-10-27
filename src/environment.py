"""
Cliff Walking Environment Utilities & Constants.
"""

from typing import Optional, Tuple
import gymnasium as gym

# Environment Constants
GRID_ROWS = 4
GRID_COLS = 12
GRID_SHAPE = (GRID_ROWS, GRID_COLS)

NUM_STATES = GRID_ROWS * GRID_COLS  # 48
NUM_ACTIONS = 4

START_STATE = 36
GOAL_STATE = 47
CLIFF_STATES = set(range(37, 47))

ACTION_NAMES = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
}

ACTION_SYMBOLS = {
    0: "↑",
    1: "→",
    2: "↓",
    3: "←",
}


def make_env(render_mode: Optional[str] = None) -> gym.Env:
    """
    Factory to create a Gymnasium CliffWalking environment.

    Args:
        render_mode: "human", "rgb_array", or None.

    Returns:
        gym.Env instance.
    """
    return gym.make("CliffWalking-v1", render_mode=render_mode)


def state_to_coord(state: int) -> Tuple[int, int]:
    """Convert flat state index [0..47] to 2D grid (row, col)."""
    return divmod(state, GRID_COLS)


def coord_to_state(row: int, col: int) -> int:
    """Convert 2D grid coordinates (row, col) to flat state index [0..47]."""
    return row * GRID_COLS + col
