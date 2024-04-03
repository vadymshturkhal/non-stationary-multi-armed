from collections import deque
import numpy as np
import torch
from game_stats import GameStats


from model import Linear_QNet, TDZeroTrainer
from settings import BET, END_MULTIPLIER, HIDDEN_LAYER_SIZE1, HIDDEN_LAYER_SIZE2, INPUT_LAYER_SIZE, MAX_MEMORY, MIN_POINTS_MULTIPLIER, MODEL_FOLDER, START_POINT


class Agent:
    """
    A class representing an agent that interacts with a stationary k-armed bandit environment.

    The agent uses an epsilon-greedy strategy for action selection. This means that with
    probability epsilon, the agent will explore and select an action at random. With
    probability 1 - epsilon, the agent will exploit its current knowledge and select the
    action with the highest estimated value.

    Attributes:
        k (int): The number of actions, i.e., the arms of the bandit.
        epsilon (float): The probability of selecting a random action; in [0, 1].
        Q (ndarray): The estimated value of each action initialized to zeros.
        N (ndarray): The count of the number of times each action has been selected.

    Methods:
        choose_action(): Selects an action using the epsilon-greedy strategy.
        update_estimates(action, reward): Updates the estimated values (Q) based on the received reward.
    """

    def __init__(self, k, epsilon=0.1):
        """
        Initializes the BanditAgent with the specified number of actions (k) and the exploration rate (epsilon).

        Parameters:
            k (int): The number of bandit arms.
            epsilon (float): The probability of exploration (choosing a random action).
        """
        self.k = k
        self.epsilon = epsilon
        self.points = START_POINT
        self.rewards = []

        # Initialize estimates of action values and action counts
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def choose_action(self) -> int:
        """
        Selects an action using the epsilon-greedy strategy.

        With probability epsilon, a random action is chosen. Otherwise, the action with the highest
        estimated value is selected.

        Returns:
            int: The index of the selected action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.Q))  # Explore: choose a random action
        else:
            return np.argmax(self.Q)  # Exploit: choose the best current action

    def update_estimates(self, action, reward):
        """
        Updates the action value estimates (Q) for a specific action based on the received reward.

        Parameters:
            action (int): The index of the action taken.
            reward (float): The reward received from taking the action.
        """
        self.N[action] += 1
        #  Update the action value estimate with the incremental sample-average formula
        self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])
    
    def update_points(self, bet, reward):
        self.points += reward - bet
        self.rewards.append(self.points)
        if self.points >= START_POINT * END_MULTIPLIER:
            return True

        if self.points <= 0:
            return True
        
        return False

class NonStationaryAgent(Agent):
    def __init__(self, k, epsilon, alpha):
        super().__init__(k, epsilon)
        self.alpha = alpha
        self.sigma = np.zeros(k)  # For the unbiased constant-step-size
    
    def choose_action(self):
        return super().choose_action()

    # Update the estimates of action values
    def update_estimates(self, action, reward):
        # Update the unbiased constant-step-size parameter
        self.sigma[action] += self.alpha * (1 - self.sigma[action])
        # Calculate the step size
        beta = self.alpha / self.sigma[action]
        # Update the estimate
        self.N[action] += 1
        self.Q[action] += beta * (reward - self.Q[action])

class NonStationaryAgentBet(NonStationaryAgent):
    def __init__(self, k, epsilon, alpha):
        super().__init__(k, epsilon, alpha)
        self.all_bet = BET
        self._available_bet = BET
    
    def _update_available_actions(self):
        available_bet = []
        for bet in self.all_bet:
            if self.points >= bet:
                available_bet.append(bet)
        self._available_bet = available_bet

    def choose_action(self):
        self._update_available_actions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self._available_bet))
        else:
            Q_available = self.Q[:len(self._available_bet)]
            arg_max = np.argmax(Q_available)
            return arg_max

    # Update the estimates of action values
    def update_estimates(self, action, reward):
        # Update the unbiased constant-step-size parameter
        self.sigma[action] += self.alpha * (1 - self.sigma[action])
        # Calculate the step size
        beta = self.alpha / self.sigma[action]
        # Update the estimate
        self.N[action] += 1
        self.Q[action] += beta * (reward - self.Q[action])

class NonStationaryAgentUCB(Agent):
    def __init__(self, k, alpha, c=2):
        """
        Initializes the agent with the specified number of actions (k), the step-size parameter (alpha),
        and the confidence level (c) for the UCB calculation.

        Parameters:
            k (int): The number of bandit arms.
            alpha (float): The step size for updating estimates.
            c (float): The confidence level for the UCB exploration term.
        """
        super().__init__(k)
        self.alpha = alpha
        self.c = c
        self.total_steps = 0

    def choose_action(self) -> int:
        """
        Selects an action using the Upper Confidence Bound (UCB) strategy.

        Returns:
            int: The index of the selected action.
        """
        self.total_steps += 1
        if np.min(self.N) == 0:  # To ensure each action is tried at least once
            return np.argmin(self.N)
        ucb_values = self.Q + self.c * np.sqrt((2 * np.log(self.total_steps)) / self.N)
        return np.argmax(ucb_values)
