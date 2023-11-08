#!/usr/bin/python3

from torch import nn
import torch
from torch.distributions import Bernoulli

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaselineNetwork(nn.Module):
    """
    A network which predicts the average reward observed
    during a markov decision-making process.
    Weights are updated w.r.t. the mean squared error between
    its prediction and the observed reward.
    """

    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(input_size, output_size)
        #input_size = nhid+1,output = 1

    def forward(self, x):
        b = self.fc(x.detach())
        return b

class Controller(nn.Module):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, ninp, nout):
        super(Controller, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(ninp, nout)  # Optimized w.r.t. reward
        #ninp = nhid+1,nout=1

    def forward(self, x):
        probs = torch.sigmoid(self.fc(x)) #the probability of halt in current stat

        probs = (1-self._epsilon)*probs + self._epsilon*torch.FloatTensor([0.05]).to(device)  # Explore/exploit
        #add Dirichlet Noise when training in favour of exploration

        m = Bernoulli(probs=probs)
        action = m.sample() # sample an action,action=1or0,the probability of action=1 is probs
        log_pi = m.log_prob(action) # compute log probability of sampled action
        #the probability of sampled action_A according to Poliy Pi when stat is S

        return action.squeeze(0), log_pi.squeeze(0), -torch.log(probs).squeeze(0)
