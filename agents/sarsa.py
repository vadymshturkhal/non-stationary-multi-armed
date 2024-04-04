import torch
import numpy as np

import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import BET, END_MULTIPLIER, MIN_POINTS_MULTIPLIER, MODEL_FOLDER, START_POINT
from settings import INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE1, HIDDEN_LAYER_SIZE2
from game_stats import GameStats
from model import Linear_QNet, SARSATrainer


class SARSA():
    def __init__(self, game, alpha=0.1, epsilon=0.1, gamma=0, is_load_weights=False):
        self.game = game
        self.epsilon = epsilon

        self.alpha = alpha
        self.gamma = gamma

        self._upper_bound = START_POINT * END_MULTIPLIER
        self._lower_bound = START_POINT * MIN_POINTS_MULTIPLIER

        self.model = Linear_QNet(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE1, HIDDEN_LAYER_SIZE2, len(BET))
        self.trainer = SARSATrainer(self.model, lr=self.alpha, gamma=self.gamma)
        self._model_filename = MODEL_FOLDER + 'sarsa_on_policy.pth'

        # Load the weights onto the CPU or GPU
        if is_load_weights:
            checkpoint = torch.load(self._model_filename)
            self._hands_played = checkpoint['hands_played']
            self._win_times = checkpoint['win_times']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

    # Update the estimates of action values
    def update_episode_estimates(self, states, actions, rewards):
        episode_loss = [0]
        for i in range(1, len(states)):
            prev_state = states[i - 1]
            prev_action = actions[i - 1]
            reward = rewards[i - 1]
            state = states[i]
            action = actions[i]
            loss = self.trainer.train_step(prev_state, prev_action, reward, state, action, done=False)
            episode_loss.append(loss)
        return episode_loss

    def choose_action(self, state):
        """Epsilon-greedy policy implementation"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(BET))
        else:
            state = torch.tensor(state, dtype=torch.float)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def save(self, stat_class: GameStats):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hands_played': stat_class.hands_played,
            'win_times': stat_class.times_win,
            }, self._model_filename)
