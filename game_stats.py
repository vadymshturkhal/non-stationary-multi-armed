import numpy as np
from collections import deque

from settings import BET


class GameStats:
    def __init__(self) -> None:
        self.hands_played = 0
        self._total_score = 0
        self._last_reward = 0
        self._win_bet = np.zeros(len(BET))
        self._times_choose_bet = np.full(len(BET), np.finfo(np.float32).tiny)
        self._prob_bet = np.zeros(len(BET))
        self._consecutive_rewards = deque([0]*16,maxlen=16)
        self._consecutive_rewards_multipliers = deque([0]*16,maxlen=16)

    @property
    def times_win(self):
        return np.sum(self._win_bet)

    def add_hand_reward(self, bet_points, bet, reward):
        self.hands_played += 1
        self._total_score += reward
        self._times_choose_bet[bet]+= 1

        if reward > 0:
            self._win_bet[bet] += 1

        self._last_reward = reward
        self._consecutive_rewards.append(reward)
        self._consecutive_rewards_multipliers.append(reward / bet_points)

        self._prob_bet = self._win_bet / self._times_choose_bet

    def reset(self):
        self.hands_played = 0
        self._total_score = 0
        self._last_reward = 0
        self._win_bet.clear()
        self._times_choose_bet.clear()
        self._prob_bet.clear()
        self._consecutive_rewards.clear()
        self._consecutive_rewards_multipliers.clear()

    def get_state(self, points):
        if self._last_reward > 0:
            recent_win_loss_indicator = 1
        elif self._last_reward == 0:
            recent_win_loss_indicator = 0
        else:
            recent_win_loss_indicator = -1

        # print(self._consecutive_rewards)
        # print(self._consecutive_rewards_multipliers)
        # print(self._win_bet)
        # print(self._times_choose_bet)
        # print(self._prob_bet)
        # print()

        state = np.array([
            points,
            recent_win_loss_indicator,
            *self._consecutive_rewards,
            *self._consecutive_rewards_multipliers,
            *self._win_bet,
            *self._times_choose_bet,
            *self._prob_bet,
        ])
        return state
