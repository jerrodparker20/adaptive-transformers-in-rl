import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(DQN, self).__init__()
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
        action = torch.argmax(q_values).reshape(-1)
        if torch.rand(1) > epsilon:
            # then take the argmax action
            return action
        else:
            # else take a random exploration action
            return torch.randint(low=0, high=self.output_layer.out_features, size=(1,))
