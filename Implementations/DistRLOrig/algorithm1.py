
import torch



'''
Improvements to make:

Make sure we're running quantile_net with large batch to improve efficiency


'''


class DistRLAgent:

    def __init__(self,quantile_net,N,k,discount=1):
        self.quantile_net = quantile_net
        self.N = N   #number of quantiles to use
        self.k = k   #distance from origin to smooth in huber loss (see equation 9: https://arxiv.org/pdf/1710.10044.pdf)
        self.discount = discount
        self.tau = torch.arange(0.0,1.0,1/(self.N+1))
        self.tau_hat = (self.tau[1:]-self.tau[:-1])/2

    '''
    Compute the quantile regression loss over several examples to improve stability. 
    '''
    def compute_loss(self,x,a,r,x_prime,theta_xprime,theta_x):
        #first computing this for a single example

        #get Q(x,a) for all a then take arg max over the actions to get next action
        Q = theta_x.mean(axis=1)
        a_prime = Q.argmax()

        #now detach from computation graph
        with torch.no_grad():
            theta_targ = r + self.discount * theta_xprime[a_prime,:]

        #now compute quantile regression loss
        u = theta_targ.view(1,-1) - theta_x.view(-1,1) #subtract all of theta_x from each element of theta_targ

        term1 = torch.abs(self.tau_hat.view(-1,1)-(u<0))
        term2 = self.k*(torch.abs(u) - k/2)
        mask = (torch.abs(u) < self.k)
        term2[mask] = (u[mask]**2) / 2
        loss = torch.sum(torch.abs(self.tau_hat.view(-1,1)-(u<0)) * term2)
        
        return loss

    def compute_loss_multi(self,x,a,r,x_prime,theta_xprime,theta_x):

        pass


