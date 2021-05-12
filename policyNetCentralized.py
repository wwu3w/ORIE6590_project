import torch
from torch import nn
from copy import deepcopy
import numpy as np
#device = "cuda" if torch.cuda.is_available() else "cpu"
class PolicyNet(nn.Module):
    def __init__(self, env):
        super(PolicyNet, self).__init__()
        self.env = deepcopy(env)
        input_size = 1 + env.R * env.R + env.R * (env.tau_d + env.L) # time, passenger state, car state
        self.embed = nn.Embedding(env.time_horizon + 1, 6)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size + 5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n),
            nn.Softmax(dim=0)
        )#policy network
        self.output_size = env.action_space.n
        self.epsilon = 0.05


    def forward(self, x):
        time = x[0].long()
        #print("Time", time)
        time_embed = self.embed(time)
        input = torch.cat((time_embed, x[1:]), 0)
        return self.linear_relu_stack(input)

    def evalCost(self, state_input, pred, R, Act, Prob, valuefnc):
        datalength = len(R)
        Act = Act.to(torch.long)
        vals = valuefnc(state_input) * valuefnc.scale
        action_prs = pred[torch.arange(datalength), Act]
        ratio = torch.div(action_prs, Prob)
        adv = R[1:datalength] + torch.reshape(torch.transpose(vals[0:datalength-1] - vals[1:datalength], 0, 1), (datalength-1,))
        tot_cost = torch.minimum(torch.clamp(ratio[1:datalength], 1 - self.epsilon, 1 + self.epsilon)*adv, ratio[1:datalength]*adv)
        return -torch.sum(tot_cost/valuefnc.dataset_size)
















