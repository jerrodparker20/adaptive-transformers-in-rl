
import torch



'''
Improvements to make:

Make sure we're running quantile_net with large batch to improve efficiency


'''


class DistMCTS:

    def __init__(self,quantile_net,N,k,discount=1):
        self.quantile_net = quantile_net
        self.N = N   #number of quantiles to use
        self.k = k   #distance from origin to smooth in huber loss (see equation 9: https://arxiv.org/pdf/1710.10044.pdf)
        self.discount = discount

    '''
    Compute the quantile regression loss over several examples to improve stability. 
    '''
    def compute_loss(self,x,a,r,x_prime,theta_xprime,theta_x):
        #first computing this for a single example

        #get Q(x,a) for all a then take arg max over the actions
        Q = theta_x.mean(axis=1)



        Q_prime = theta_xprime.mean()

