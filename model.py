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

class TDZeroTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
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

class SARSATrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, next_action, done: bool):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        next_action = torch.tensor(next_action, dtype=torch.long)

        # (n, )
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_action = torch.unsqueeze(next_action, 0)

        q_values = self.model(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.model(next_state).gather(1, next_action.unsqueeze(-1)).squeeze(-1)
        expected_q_values = reward + self.gamma * next_q_values * (1 - float(done))

        loss = self.criterion(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class QLearningTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        # Reshape for batch size 1, if necessary
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)

        # Compute current Q values (Q(s,a))
        current_q_values = self.model(state).gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Compute next Q values (max Q(s',a))
        next_q_values = self.model(next_state).detach().max(1)[0]

        # Compute the target of the current Q values
        target_q_values = reward + (1 - done) * self.gamma * next_q_values

        # Calculate loss
        loss = self.criterion(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
