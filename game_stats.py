from settings import BET


class GameStats:
    def __init__(self, game) -> None:
        self.game = game
        self._hands_played = 0
        self._win_bet = [0] * len(BET)
        self._times_choose_bet = [0] * len(BET)
        self._prob_bet = [0] * len(BET)

    def add_hand_reward(self, action, reward):
        self._win_times += 1 if reward > 0 else 0


    def reset(self):
        self._hands_played = 0
        self._win_times = 0
        self._win_bet.clear()
        self._times_choose_bet.clear()
        self._prob_bet.clear()