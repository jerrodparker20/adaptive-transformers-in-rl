

import torch.nn as nn
import torch


'''

TO DO:
Add in layer normalization to transformer encoder
Remember at test time since we already have a well trained model, I think we don't send in 
extra states anymore when computing the Q-values (can try both). If do send extras then need to make 
sure the dropout is turned off in transformer encoder but not in the first embedding layers. 

Right now dropout scales the outputs during training, is this what we want for those first layers?
Check that dropout implementation in pytorch has different dropout per element in the batch #$$$$$$$$$$


DQN now takes as input a set of states, then feeds each one through the embedder B times,
which will help encoder uncertainty of the embeddings, and then feed the combined results through 
transformer encoder
'''

class CartPoleEmbedder(nn.Module):
    def __init__(self,dropout, B,input_size, embedding_size):
        '''
        :param B: Number of times we embed each state (with dropout each time)

        '''

        super(CartPoleEmbedder, self).__init__()
        self.B = B
        self.dropout_p = dropout
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.layer3 = nn.Linear(embedding_size,embedding_size)

        #now need to combine the B copies of the elements
        #Can start by using just linear combo then move to nonlinear combo


        '''
        self.layer3 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU()
        )
        '''

    def forward(self,input,is_training=True):
        #want to now stack B copies of input on top of eachother
        #Batch dim is dim 0
        input = torch.cat(self.B*[input])

        #The dropout implementation in pytorch applies dropout differently per element in batch
        #which is what we want

        hidden = self.layer1(input)
        #hidden = self.layer2(hidden)
        return self.layer3(hidden)



class TransformerDqn(nn.Module):

    def __init__(self,embedder,embedder_params,encoder_layer_params,num_encoder_layers,output_size):
        '''
        :param embedder: module to embed the states
        :param output_size: number of actions we can choose
        '''

        dropout = embedder_params['dropout']

        super(TransformerDqn, self).__init__()
        self.embedder = embedder(**embedder_params)
        self.encoder_layer = nn.TransformerEncoderLayer(**encoder_layer_params)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(encoder_layer_params['d_model'],output_size)

    def forward(self,input):
        '''
        :param input: matrix of state vectors (last column will contain state of interest)
        :return: vector of Q values for each action
        '''

        embedding = self.embedder(input)
        embedding = self.encoder(embedding)
        return self.output_layer(embedding)




