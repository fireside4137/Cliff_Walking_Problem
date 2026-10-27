"""
Reinforcement Learning Agents for Cliff Walking.
Implements Tabular Q-Learning, SARSA, and Expected SARSA algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Abstract Base Class for Tabular Reinforcement Learning Agents.
    """

    def __init__(
        self,
        num_states: int = 48,
        num_actions: int = 4,
        alpha: float = 0.5,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        """
        Initialize the RL Agent.

        Args:
            num_states: Total number of discrete states.
            num_actions: Total number of discrete actions.
            alpha: Learning rate (step size), default 0.5.
            gamma: Discount factor, default 0.99.
            epsilon: Exploration probability for epsilon-greedy policy, default 0.1.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float64)

    def act(self, state: int, evaluate: bool = False) -> int:
        """
        Select an action using the epsilon-greedy policy.

        Args:
            state: The current state index.
            evaluate: If True, acts purely greedily (epsilon=0).

        Returns:
            The selected action index.
        """
        eps = 0.0 if evaluate else self.epsilon
        if np.random.random() < eps:
            return np.random.randint(self.num_actions)

        # Break ties randomly among max Q-values to avoid directional bias
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)

    @abstractmethod
    def update(self, *args, **kwargs) -> float:
        """
        Update the Q-table given experience.
        Must be implemented by subclasses.
        """
        pass

    def get_policy(self) -> np.ndarray:
        """
        Return the greedy deterministic policy: state -> best action.
        """
        return np.argmax(self.q_table, axis=1)

    def save(self, filepath: str) -> None:
        """Save Q-table to a NumPy .npy file."""
        np.save(filepath, self.q_table)

    def load(self, filepath: str) -> None:
        """Load Q-table from a NumPy .npy file."""
        self.q_table = np.load(filepath)


class QLearningAgent(BaseAgent):
    """
    Q-Learning (Watkins, 1989): Off-Policy Temporal Difference Control.
    Directly approximates the optimal action-value function Q* independent
    of the policy being followed.
    """

    def __init__(
        self,
        num_states: int = 48,
        num_actions: int = 4,
        alpha: float = 0.5,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        super().__init__(num_states, num_actions, alpha, gamma, epsilon)
        self.name = "Q-Learning"

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """
        Q-learning Bellman optimality update:
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
        """
        td_target = reward
        if not done:
            td_target += self.gamma * np.max(self.q_table[next_state])

        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return td_error


class SARSAAgent(BaseAgent):
    """
    SARSA (Rummery & Niranjan, 1994): On-Policy Temporal Difference Control.
    Learns the action-value function of the epsilon-greedy policy actually followed,
    making it risk-averse in stochastic or exploratory cliff navigation.
    """

    def __init__(
        self,
        num_states: int = 48,
        num_actions: int = 4,
        alpha: float = 0.5,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        super().__init__(num_states, num_actions, alpha, gamma, epsilon)
        self.name = "SARSA"

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ) -> float:
        """
        SARSA Bellman expectation update:
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        """
        td_target = reward
        if not done:
            td_target += self.gamma * self.q_table[next_state, next_action]

        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return td_error


class ExpectedSARSAAgent(BaseAgent):
    """
    Expected SARSA (van Seijen et al., 2009):
    Computes the mathematical expectation over all possible next actions
    under the epsilon-greedy policy, reducing variance compared to SARSA.
    """

    def __init__(
        self,
        num_states: int = 48,
        num_actions: int = 4,
        alpha: float = 0.5,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        super().__init__(num_states, num_actions, alpha, gamma, epsilon)
        self.name = "Expected SARSA"

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """
        Expected SARSA update:
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * E_pi[Q(S', .)] - Q(S, A)]
        """
        td_target = reward
        if not done:
            # Policy probabilities for epsilon-greedy
            q_next = self.q_table[next_state]
            best_action = np.argmax(q_next)
            action_probs = np.full(self.num_actions, self.epsilon / self.num_actions)
            action_probs[best_action] += 1.0 - self.epsilon

            expected_q = np.sum(action_probs * q_next)
            td_target += self.gamma * expected_q

        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return td_error
