import abc
from typing import List

import numpy as np

from bandit import Bandit


class BaseMABSimulator(abc.ABC):
    def __init__(self, bandits: List[Bandit], steps: int = 1000):
        self.bandits = bandits
        self.steps = steps
        self._reset_params()

    def _reset_params(self) -> None:
        self.Q = self.init_expected_rewards()
        self.N = np.zeros(shape=(len(self.bandits)))

    def init_expected_rewards(self) -> np.ndarray:
        return np.zeros(shape=(len(self.bandits)))

    @abc.abstractmethod
    def select_bandit(self) -> int:
        """Returns index of selected bandit, using a particular strategy"""

    @abc.abstractmethod
    def update_expected_reward(self, R: float, bandit_index: int) -> None:
        """Updates the expected reward for a bandit"""

    def _simulate_one_run(self) -> np.ndarray:
        self._reset_params()
        average_rewards = []
        for _ in range(self.steps):
            bandit_index = self.select_bandit()
            obtained_reward = self.bandits[bandit_index].generate_reward()
            self.N[bandit_index] += 1
            self.update_expected_reward(obtained_reward, bandit_index)
            average_rewards.append(self.Q.max())
        return average_rewards

    def simulate_multiple_runs(self, num_runs: int = 1000) -> np.ndarray:
        all_rewards = []
        for _ in range(num_runs):
            av_rewards = self._simulate_one_run()
            all_rewards.append(av_rewards)
        all_rewards = np.vstack(all_rewards)
        return all_rewards.mean(axis=0)


class GreedyMABSimulator(BaseMABSimulator):
    """Greedy simulator - selects the bandit with the current highest expected reward at each step"""

    def select_bandit(self) -> int:
        max_reward = np.max(self.Q)
        mask = self.Q == max_reward
        inds = np.nonzero(mask)[0]
        bandit_index = np.random.choice(inds)
        return bandit_index

    def update_expected_reward(self, R: float, bandit_index: int) -> None:
        Q  = self.Q
        prev_q = Q[bandit_index]
        Q[bandit_index] = prev_q + (R - prev_q) / self.N[bandit_index]

    def __repr__(self) -> str:
        return "Greedy MAB simulator"


class EpsilonGreedyMABSimulator(GreedyMABSimulator):
    """Same as greedy, but with more inclanation to explore"""

    def __init__(self, bandits: List[Bandit], epsilon: float = 0.1, steps: int = 1000):
        super().__init__(bandits, steps)
        self.epsilon = epsilon

    def select_bandit(self) -> int:
        greedy_step = np.random.random() > self.epsilon
        if greedy_step:
            return super().select_bandit()
        return np.random.randint(len(self.bandits))

    def __repr__(self) -> str:
        return f"Epsilon-greedy MAB simulator(epsilon={self.epsilon})"
