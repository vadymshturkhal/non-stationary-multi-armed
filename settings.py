# Data filename
nonstationary_bandit_data_average_reward = './data/nonstationary_bandit_average_reward.csv'

# Model
MODEL_FOLDER = './model/'

# Bet
ACTION_COST = 1
START_POINT = 2000
MIN_POINTS_MULTIPLIER = 0.2
END_MULTIPLIER = 2.5
BET = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# Batch
MAX_MEMORY = 1024

# Model
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 1e-5
MIN_EPSILON = 0.1

# Model layers
INPUT_LAYER_SIZE = 61
HIDDEN_LAYER_SIZE1 = 256
HIDDEN_LAYER_SIZE2 = 256

# Rewards
REWARD_WIN = 1
REWARD_LOOSE = -1
