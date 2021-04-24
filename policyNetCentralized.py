import torch
import numpy as np
from torch import nn
import gym

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
class PolicyNet(nn.Module):
    def __init__(self, env):
        super(PolicyNet, self).__init__()
        self.env = env
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
        self.epsilon = 0.9


    def forward(self, x):
        return self.linear_relu_stack(x)

    def evalCost(self, state_input, pred, R, Act, Prob, valuefnc):
        datalength = len(R)
        tot_cost = 0.0
        for i in range(datalength-1):
            r = R[i] #reward
            a = Act[i] #action

            a_distribution = pred[i]
            #print(a_distribution)
            state = state_input[i]
            #print(state)
            state_next = state_input[i+1]
            val = valuefnc(state) * valuefnc.scale
            val_next = valuefnc(state_next) * valuefnc.scale
            pr = Prob[i]
            Advantfnc = r + val_next - val
            ratio = a_distribution[a]/pr
            #print("a_distribution", a_distribution)
            #print("act", a)
            #print("pr", pr)
            print("ratio", ratio)
            ratio = np.clip(ratio.item(), 1 - self.epsilon, 1 + self.epsilon)

            tot_cost += ratio * Advantfnc
        return tot_cost/valuefnc.dataset_size









