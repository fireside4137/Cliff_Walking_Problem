"""
Training, Evaluation, and Benchmarking Harness for Cliff Walking Agents.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import gymnasium as gym

from .agents import BaseAgent, SARSAAgent
from .environment import make_env, CLIFF_STATES, GOAL_STATE


def train(
    agent: BaseAgent,
    env: Optional[gym.Env] = None,
    episodes: int = 500,
    max_steps_per_episode: int = 1000,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Train an agent in the CliffWalking environment.

    Args:
        agent: BaseAgent instance (QLearningAgent, SARSAAgent, etc.)
        env: gym.Env instance (created automatically if None).
        episodes: Total number of training episodes.
        max_steps_per_episode: Maximum steps allowed per episode before truncation.
        seed: Random seed for reproducibility.
        verbose: Print progress every 50 episodes.

    Returns:
        Dictionary containing training history:
            - 'rewards': List of total rewards per episode.
            - 'lengths': List of step counts per episode.
            - 'cliff_falls': List of cliff drops per episode.
    """
    close_env = False
    if env is None:
        env = make_env()
        close_env = True

    if seed is not None:
        np.random.seed(seed)

    history = {
        "rewards": [],
        "lengths": [],
        "cliff_falls": [],
    }

    is_sarsa = isinstance(agent, SARSAAgent)

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode if seed is not None else None)
        done = False
        total_reward = 0.0
        steps = 0
        cliff_falls = 0

        action = agent.act(state) if is_sarsa else None

        while not done and steps < max_steps_per_episode:
            if not is_sarsa:
                action = agent.act(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Track cliff falls (reward is -100 when falling into the cliff)
            if reward <= -100:
                cliff_falls += 1

            if is_sarsa:
                next_action = agent.act(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                action = next_action
            else:
                agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        history["rewards"].append(total_reward)
        history["lengths"].append(steps)
        history["cliff_falls"].append(cliff_falls)

        if verbose and (episode + 1) % 50 == 0:
            avg_r = np.mean(history["rewards"][-50:])
            avg_falls = np.mean(history["cliff_falls"][-50:])
            print(
                f"[{agent.name}] Episode {episode + 1:4d}/{episodes} | "
                f"Avg Reward (last 50): {avg_r:7.1f} | "
                f"Avg Cliff Falls: {avg_falls:4.2f} | Steps: {steps:3d}"
            )

    if close_env:
        env.close()

    return history


def evaluate(
    agent: BaseAgent,
    render: bool = False,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """
    Evaluate the agent under a pure greedy policy (no exploration).

    Args:
        agent: Trained BaseAgent.
        render: If True, renders via Gymnasium's human mode.
        max_steps: Maximum steps before cutting off.

    Returns:
        Dictionary containing evaluation results:
            - 'path': List of states visited.
            - 'actions': List of actions taken.
            - 'total_reward': Cumulative reward earned.
            - 'steps': Total steps.
            - 'reached_goal': True if terminal state reached.
    """
    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)

    state, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    path = [state]
    actions_taken = []

    while not done and steps < max_steps:
        action = agent.act(state, evaluate=True)
        actions_taken.append(action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
        path.append(state)
        total_reward += reward
        steps += 1

    reached_goal = (state == GOAL_STATE)
    env.close()

    return {
        "path": path,
        "actions": actions_taken,
        "total_reward": total_reward,
        "steps": steps,
        "reached_goal": reached_goal,
    }


def benchmark(
    agent_classes: List[type],
    episodes: int = 500,
    runs: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Benchmark multiple algorithms across multiple random seeds
    to produce mean and standard error of rewards and cliff falls.

    Args:
        agent_classes: List of agent classes to benchmark.
        episodes: Number of episodes per run.
        runs: Number of independent random seeds.
        seed: Initial random seed.

    Returns:
        Nested dict of {agent_name: {'mean_rewards': ..., 'std_rewards': ..., ...}}
    """
    results = {}

    for cls in agent_classes:
        dummy = cls()
        name = dummy.name
        print(f"--> Running benchmark for {name} ({runs} runs x {episodes} episodes)...")

        all_rewards = np.zeros((runs, episodes))
        all_falls = np.zeros((runs, episodes))

        for r in range(runs):
            current_seed = seed + r * 100
            agent = cls()
            env = make_env()
            history = train(agent, env=env, episodes=episodes, seed=current_seed, verbose=False)
            env.close()

            all_rewards[r] = history["rewards"]
            all_falls[r] = history["cliff_falls"]

        results[name] = {
            "mean_rewards": np.mean(all_rewards, axis=0),
            "std_rewards": np.std(all_rewards, axis=0) / np.sqrt(runs),
            "mean_falls": np.mean(all_falls, axis=0),
            "std_falls": np.std(all_falls, axis=0) / np.sqrt(runs),
            "sample_agent": agent,  # keep one trained agent for evaluation / policy visualization
        }

    return results
