#!/usr/bin/env python3
"""
Cliff Walking Reinforcement Learning: Q-Learning vs. SARSA Benchmark.

Usage examples:
    python main.py --algo all --episodes 500
    python main.py --algo q_learning --render
    python main.py --compare --runs 5
"""

import argparse
import os
import sys

from src.agents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from src.environment import make_env
from src.trainer import train, evaluate, benchmark
from src.visualizer import (
    plot_learning_curves,
    plot_policy_and_trajectory,
    plot_comparison_paths,
    print_ascii_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and evaluate RL agents on CliffWalking-v1."
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        choices=["q_learning", "sarsa", "expected_sarsa", "all"],
        help="Algorithm to train/evaluate (default: all).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes (default: 500).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Learning rate alpha (default: 0.5).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor gamma (default: 0.99).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate epsilon (default: 0.1).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of independent seeds for benchmarking (default: 5).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes in a graphical window.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run multi-run statistical benchmark and generate comparison plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating and saving plot images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("assets", exist_ok=True)

    agent_map = {
        "q_learning": QLearningAgent,
        "sarsa": SARSAAgent,
        "expected_sarsa": ExpectedSARSAAgent,
    }

    if args.compare or args.algo == "all":
        print("=" * 60)
        print(" Cliff Walking RL Benchmark: Q-Learning vs. SARSA ")
        print("=" * 60)

        classes_to_run = [QLearningAgent, SARSAAgent, ExpectedSARSAAgent]
        bench_results = benchmark(
            classes_to_run,
            episodes=args.episodes,
            runs=args.runs,
        )

        print("\n--- Benchmark Complete! Evaluating Greedy Policies ---")
        paths = {}
        for cls in classes_to_run:
            agent = bench_results[cls().name]["sample_agent"]
            eval_res = evaluate(agent, render=args.render)
            paths[agent.name] = eval_res["path"]

            print(
                f"[{agent.name}] Greedy Path Length: {eval_res['steps']} steps | "
                f"Total Reward: {eval_res['total_reward']:.0f} | "
                f"Reached Goal: {eval_res['reached_goal']}"
            )
            print_ascii_policy(agent)

            if not args.no_plots:
                slug = agent.name.lower().replace(" ", "_")
                plot_policy_and_trajectory(
                    agent,
                    eval_res["path"],
                    save_path=f"assets/{slug}_policy.png",
                )

        if not args.no_plots:
            plot_learning_curves(bench_results, save_path="assets/learning_curves.png")
            if "Q-Learning" in paths and "SARSA" in paths:
                plot_comparison_paths(
                    paths["Q-Learning"],
                    paths["SARSA"],
                    save_path="assets/path_comparison.png",
                )
        print("\nAll tasks completed successfully. Check the 'assets/' folder for visualizations.")

    else:
        agent_cls = agent_map[args.algo]
        agent = agent_cls(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
        print(f"--> Training {agent.name} for {args.episodes} episodes...")

        env = make_env()
        train(agent, env=env, episodes=args.episodes, verbose=True)
        env.close()

        print(f"\n--> Evaluating {agent.name} with greedy policy...")
        eval_res = evaluate(agent, render=args.render)
        print(
            f"Evaluation result: Steps: {eval_res['steps']}, "
            f"Reward: {eval_res['total_reward']}, Reached Goal: {eval_res['reached_goal']}"
        )
        print_ascii_policy(agent)

        if not args.no_plots:
            slug = agent.name.lower().replace(" ", "_")
            plot_policy_and_trajectory(
                agent,
                eval_res["path"],
                save_path=f"assets/{slug}_policy.png",
            )
            print(f"Saved policy visualization to assets/{slug}_policy.png")


if __name__ == "__main__":
    main()
