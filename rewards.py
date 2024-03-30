import numpy as np

from settings import BET


class Rewards:
    def __init__(self, k:int):
        self.true_reward_probabilities = np.random.rand(k)

    # Simulate pulling the bandit's lever
    def get_reward(self, action):
        return 1 if (np.random.rand() < self.true_reward_probabilities[action]) else 0

    def change_reward_probabilities(self):
        self.true_reward_probabilities += np.random.normal(0, 0.01, size=self.true_reward_probabilities.shape)
        self.true_reward_probabilities = np.clip(self.true_reward_probabilities, 0, 1)

class DealerRewards:
    def __init__(self):
        self.bet = BET
        self._multiply = [0, 1, 2, 4, 8, 16, 32]
        self._probabilities = [0.7, 0.14, 0.08, 0.05, 0.02, 0.009, 0.001]

    # Simulate pulling the bandit's lever
    def get_reward(self, bet):
        return self.bet[bet] * np.random.choice(self._multiply, p=self._probabilities)
