import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from settings import DROPOUT_RATE, WEIGHT_DECAY


class Linear_QNet(nn.Module):
    def __init__(self, input_layer, hidden1, hidden2, output_layer):
        super().__init__()
        self.linear1 = nn.Linear(input_layer, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        self.linear3 = nn.Linear(hidden2, output_layer)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.linear3(x)
        return x

    def save(self, epoch=0, filename=None):
        torch.save({
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            }, filename)

class TDZeroTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
        
        # Get current state value
        current_v_values = self.model(state)
        
        # Get next state value
        next_v_values = self.model(next_state).detach()
        
        # If not done, use the TD(0) update rule, else use reward as the final value
        target = reward + (self.gamma * next_v_values * (1 - float(done)))
        
        self.optimizer.zero_grad()
        loss = self.criterion(current_v_values, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()
