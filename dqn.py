import torch.nn as nn
import torch.optim as optim
import gym
import torch
from torch.distributions import Categorical, Normal
from torch.nn.functional import mse_loss
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

# In[]:

class ReplayBuffer:
    def __init__(self, max_size=2000):
        self.max_size = max_size

        self.cur_states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.cur_states)

    def add(self, cur_state, action, next_state, reward, done):
        self.cur_states.append(cur_state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, sample_size=32):
        sample_transitions = {}
        if self.__len__() >= sample_size:
            # pick up only random 32 events from the memory
            indices = np.random.choice(self.__len__(), size=sample_size)
            sample_transitions['cur_states'] = torch.stack(self.cur_states)[indices]
            sample_transitions['actions'] = torch.stack(self.actions)[indices]
            sample_transitions['next_states'] = torch.stack(self.next_states)[indices]
            sample_transitions['rewards'] = torch.Tensor(self.rewards)[indices]
            sample_transitions['dones'] = torch.Tensor(self.dones)[indices]
        else:
            # if the current buffer size is not greater than 32 then pick up the entire memory
            sample_transitions['cur_states'] = torch.stack(self.cur_states)
            sample_transitions['actions'] = torch.stack(self.actions)
            sample_transitions['next_states'] = torch.stack(self.next_states)
            sample_transitions['rewards'] = torch.Tensor(self.rewards)
            sample_transitions['dones'] = torch.Tensor(self.dones)

        return sample_transitions

# In[]:

class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(DQN, self).__init__()
        self.num_actions = output_size
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output_layer(out)
        return out

    def select_action(self, current_state, epsilon=0.1):
        """
        selects an action as per the decided exploration
        :param current_state: the current state
        :param epsilon: the param for exploration, typical value = 0.1
        :return: the chosen action

        """
        q_values = self(current_state)
        action = torch.argmax(q_values)
        if torch.rand(1) > epsilon:
            # then take the argmax action
            return action
        else:
            # else take a random exploration action
            return torch.randint(low=0, high=self.num_actions+1, size=(1,)).item()