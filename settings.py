nonstationary_bandit_data_average_reward = './data/nonstationary_bandit_average_reward.csv'
ACTION_COST = 1
START_POINT = 1000
END_MULTIPLIER = 1.2
BET = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# Batch
MAX_MEMORY = 1024

# Model
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 1e-5

# Model layers
INPUT_LAYER_SIZE = 2
HIDDEN_LAYER_SIZE1 = 128
HIDDEN_LAYER_SIZE2 = 128
OUTPUT_LAYER_SIZE = 1
