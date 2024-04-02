# Data filename
nonstationary_bandit_data_average_reward = './data/nonstationary_bandit_average_reward.csv'

# Model
MODEL_FOLDER = './model/'

# Bet
ACTION_COST = 1
START_POINT = 2000
MIN_POINTS_MULTIPLIER = 0.3
END_MULTIPLIER = 2
BET = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# Batch
MAX_MEMORY = 1024

# Model
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 1e-5

# Model layers
INPUT_LAYER_SIZE = 27
HIDDEN_LAYER_SIZE1 = 256
HIDDEN_LAYER_SIZE2 = 256
