import numpy as np


class Bandit:
    def __init__(self):
        self._loc = np.random.normal(loc=0.0)

    def generate_reward(self) -> float:
        return np.random.normal(self._loc)
