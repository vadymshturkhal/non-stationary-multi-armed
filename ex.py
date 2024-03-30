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
