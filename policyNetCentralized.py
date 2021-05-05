import torch
from torch import nn
from copy import deepcopy
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
class PolicyNet(nn.Module):
    def __init__(self, env):
        super(PolicyNet, self).__init__()
        self.env = deepcopy(env)
        input_size = 1 + env.R * env.R + env.R * (env.tau_d + env.L) # time, passenger state, car state
        out_put_size = env.R * env.R
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
            nn.Softmax(dim=0)
        )#policy network
        self.output_size = env.action_space.n
        self.epsilon = 0.999


    def forward(self, x):
        return self.linear_relu_stack(x)

    def evalCost(self, state_input, pred, R, Act, Prob, valuefnc):
        datalength = len(R)
        Act = Act.to(torch.long)
        vals = valuefnc(state_input) * valuefnc.scale
        action_prs = pred[torch.arange(datalength), Act]
        ratio = torch.div(action_prs, Prob)
        #print("ratio: ", torch.minimum(torch.clamp(ratio[1:datalength], 1 - self.epsilon, 1 + self.epsilon),ratio[1:datalength]))
        #print("adv: ", (R[1:datalength] + torch.reshape(torch.transpose(vals[0:datalength-1] - vals[1:datalength], 0, 1), (datalength-1,))))
        tot_cost = torch.minimum(torch.clamp(ratio[1:datalength], 1 - self.epsilon, 1 + self.epsilon),ratio[1:datalength])  * (R[1:datalength] + torch.reshape(torch.transpose(vals[0:datalength-1] - vals[1:datalength], 0, 1), (datalength-1,)))
        return -torch.sum(tot_cost/valuefnc.dataset_size)
















