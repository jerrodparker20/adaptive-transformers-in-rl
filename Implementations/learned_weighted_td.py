
from transformerDqn import *
import gym
import torch
from dqn import DQN, ReplayBuffer
from torch.optim import Adam
from torch.nn.functional import mse_loss

#creating DQN
embedding_size = 24
dropout = 0.1
B = 32
input_size = 3
dim_feedforward = 16
nhead = 1
num_actions = 4
num_encoder_layers = 1
embedder = CartPoleEmbedder
embedding_params = {'dropout': dropout, 'B': B, 'input_size': input_size, 'embedding_size': embedding_size}
encoder_layer_params = {'d_model':embedding_size, 'nhead':nhead, 'dim_feedforward':dim_feedforward, 'dropout':dropout}
dqn = TransformerDqn(embedder=embedder,embedder_params=embedding_params,
                     encoder_layer_params=encoder_layer_params,output_size=num_actions,
                     num_encoder_layers=1)



