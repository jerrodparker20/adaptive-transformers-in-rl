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