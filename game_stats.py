import numpy as np

from settings import BET, END_MULTIPLIER, MIN_POINTS_MULTIPLIER, START_POINT


class GameStats:
    def __init__(self) -> None:
        self.hands_played = 0
        self._total_score = 0
        self._win_bet = np.zeros(len(BET))
        self._times_choose_bet = np.full(len(BET), np.finfo(np.float32).tiny)
        self._prob_bet = np.zeros(len(BET))

    @property
    def times_win(self):
        return np.sum(self._win_bet)

    def add_hand_reward(self, bet, reward):
        self.hands_played += 1
        self._total_score += reward
        self._times_choose_bet[bet]+= 1

        if reward > 0:
            self._win_bet[bet] += 1

        self._prob_bet = self._win_bet / self._times_choose_bet

    def reset(self):
        self.hands_played = 0
        self._total_score = 0
        self._win_bet.clear()
        self._times_choose_bet.clear()
        self._prob_bet.clear()

    def get_state(self):
        general_win_probability = np.sum(self._win_bet) / self.hands_played
        state = np.array([
            *self._win_bet,
            *self._times_choose_bet,
            *self._prob_bet,
        ])
        return state
