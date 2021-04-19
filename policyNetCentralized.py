import torch
import numpy as np
from torch import nn
import gym

device = "cpu"
class PolicyNet(nn.Module):
    def __init__(self, env):
        self.env = env
        input_size = 1 + env.R * env.R + env.R * (env.tau_d + env.L) # time, passenger state, car state
        out_put_size = env.R * env.R
        self.linear_rellu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n)
        )#policy network
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear_rellu_stack(x))








