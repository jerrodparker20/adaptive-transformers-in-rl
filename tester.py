

import torch.nn as nn
import torch

class Tester(nn.Module):
    def __init__(self):
        super(Tester, self).__init__()

        self.linear = nn.Linear(5,5)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,input):

        output = self.linear(input)
        print(self.dropout(output))
        return output

test = Tester()
input = torch.rand(1,5)


test.forward(input)