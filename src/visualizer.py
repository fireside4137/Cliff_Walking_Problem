"""
Visualization utilities for Cliff Walking RL policies and benchmark curves.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .environment import (
    GRID_ROWS,
    GRID_COLS,
    START_STATE,
    GOAL_STATE,
    CLIFF_STATES,
    ACTION_SYMBOLS,
    state_to_coord,
)


def moving_average(data: np.ndarray, window_size: int = 10) -> np.ndarray:
    """Calculate moving average with edge padding."""
    if len(data) < window_size:
        return data
    pad = window_size // 2
    padded = np.pad(data, (pad, pad), mode="edge")
    conv = np.convolve(padded, np.ones(window_size) / window_size, mode="valid")
    return conv[:len(data)]


def plot_learning_curves(
    benchmark_results: Dict[str, Dict[str, Any]],
    window: int = 15,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot comparative learning curves: Episodic Rewards & Cliff Falls.

    Args:
        benchmark_results: Output from trainer.benchmark()
        window: Moving average smoothing window.
        save_path: Filepath to save figure (optional).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"Q-Learning": "#E63946", "SARSA": "#1D3557", "Expected SARSA": "#2A9D8F"}

    for name, data in benchmark_results.items():
        color = colors.get(name, "#457B9D")
        mean_r = moving_average(data["mean_rewards"], window)
        std_r = moving_average(data["std_rewards"], window)
        episodes = np.arange(1, len(mean_r) + 1)

        # Reward plot
        ax1.plot(episodes, mean_r, label=name, color=color, linewidth=2)
        ax1.fill_between(episodes, mean_r - std_r, mean_r + std_r, color=color, alpha=0.15)

        # Cliff falls plot
        mean_f = moving_average(data["mean_falls"], window)
        std_f = moving_average(data["std_falls"], window)
        ax2.plot(episodes, mean_f, label=name, color=color, linewidth=2)
        ax2.fill_between(episodes, mean_f - std_f, mean_f + std_f, color=color, alpha=0.15)

    ax1.set_title("Training Reward per Episode (Smoothed)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Total Reward", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.set_title("Cliff Falls per Episode (Smoothed)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Number of Falls into Cliff", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved learning curves to: {save_path}")
    plt.close()


def plot_policy_and_trajectory(
    agent: Any,
    trajectory: Optional[List[int]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Render 4x12 grid with policy arrows, special states, and evaluation trajectory.

    Args:
        agent: Trained agent with .q_table.
        trajectory: List of visited state indices during evaluation.
        title: Plot title.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(0, GRID_ROWS)
    ax.set_aspect("equal")

    # Invert Y so row 0 is at the top
    ax.invert_yaxis()

    # Draw grid tiles
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            state = r * GRID_COLS + c

            if state == START_STATE:
                color = "#4CAF50"  # Start (Green)
                label = "START\n(S)"
            elif state == GOAL_STATE:
                color = "#FFD700"  # Goal (Gold)
                label = "GOAL\n(G)"
            elif state in CLIFF_STATES:
                color = "#E63946"  # Cliff (Red)
                label = "THE CLIFF\n(-100)"
            else:
                color = "#F1FAEE" if (r + c) % 2 == 0 else "#E9ECEF"
                label = ""

            rect = patches.Rectangle(
                (c, r), 1, 1, linewidth=1.2, edgecolor="#2B2D42", facecolor=color
            )
            ax.add_patch(rect)

            if label:
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#1D3557" if state != START_STATE else "white",
                )

            # Draw greedy policy arrow if not cliff or goal
            if state not in CLIFF_STATES and state != GOAL_STATE:
                best_action = np.argmax(agent.q_table[state])
                symbol = ACTION_SYMBOLS[best_action]
                ax.text(
                    c + 0.5,
                    r + 0.75 if label else r + 0.5,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="#1D3557",
                )

    # Draw evaluation trajectory if provided
    if trajectory and len(trajectory) > 1:
        coords = [state_to_coord(s) for s in trajectory]
        y_pts = [r + 0.5 for r, c in coords]
        x_pts = [c + 0.5 for r, c in coords]

        ax.plot(
            x_pts,
            y_pts,
            color="#2A9D8F",
            linewidth=3.5,
            linestyle="-",
            marker="o",
            markersize=6,
            label="Evaluation Path",
            zorder=5,
        )

    algo_name = getattr(agent, "name", "Agent")
    ax.set_title(title or f"{algo_name} - Learned Policy & Greedy Trajectory", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.set_xticklabels([f"Col {c}" for c in range(GRID_COLS)])
    ax.set_yticklabels([f"Row {r}" for r in range(GRID_ROWS)])

    if trajectory:
        ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved policy plot to: {save_path}")
    plt.close()


def plot_comparison_paths(
    q_path: List[int],
    sarsa_path: List[int],
    save_path: Optional[str] = None,
) -> None:
    """
    Render both Q-Learning (optimal/risky) and SARSA (safe detour) paths
    on the same grid for immediate visual contrast.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(0, GRID_ROWS)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            state = r * GRID_COLS + c
            if state == START_STATE:
                color = "#4CAF50"
                label = "START"
            elif state == GOAL_STATE:
                color = "#FFD700"
                label = "GOAL"
            elif state in CLIFF_STATES:
                color = "#D62828"
                label = "CLIFF"
            else:
                color = "#F8F9FA" if (r + c) % 2 == 0 else "#E9ECEF"
                label = ""

            rect = patches.Rectangle((c, r), 1, 1, edgecolor="#ADB5BD", facecolor=color)
            ax.add_patch(rect)
            if label:
                ax.text(c + 0.5, r + 0.5, label, ha="center", va="center", fontsize=9, fontweight="bold")

    # Q-Learning path (red offset slightly up)
    q_coords = [state_to_coord(s) for s in q_path]
    qx = [c + 0.5 for r, c in q_coords]
    qy = [r + 0.45 for r, c in q_coords]
    ax.plot(qx, qy, color="#E63946", linewidth=3.5, marker="s", markersize=6, label=f"Q-Learning Path ({len(q_path)-1} steps, Optimal/Risky)")

    # SARSA path (blue offset slightly down)
    s_coords = [state_to_coord(s) for s in sarsa_path]
    sx = [c + 0.5 for r, c in s_coords]
    sy = [r + 0.55 for r, c in s_coords]
    ax.plot(sx, sy, color="#1D3557", linewidth=3.5, linestyle="--", marker="o", markersize=6, label=f"SARSA Path ({len(sarsa_path)-1} steps, Safe Detour)")

    ax.set_title("Optimal vs. Safe Path: Q-Learning vs. SARSA in Cliff Walking", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=11, frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved path comparison plot to: {save_path}")
    plt.close()


def print_ascii_policy(agent: Any) -> None:
    """Print ASCII representation of the policy to stdout."""
    print(f"\n=== Learned Policy: {agent.name} ===")
    for r in range(GRID_ROWS):
        row_str = []
        for c in range(GRID_COLS):
            state = r * GRID_COLS + c
            if state == START_STATE:
                row_str.append(" S ")
            elif state == GOAL_STATE:
                row_str.append(" G ")
            elif state in CLIFF_STATES:
                row_str.append(" C ")
            else:
                best_action = np.argmax(agent.q_table[state])
                row_str.append(f" {ACTION_SYMBOLS[best_action]} ")
        print("|" + "|".join(row_str) + "|")
    print("=" * 45 + "\n")
