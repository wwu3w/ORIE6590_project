import torch
import numpy as np
from torch import nn
import gym
device = "cpu"
class valueEstimator(nn.Module):
    def __init__(self, env):
        self.env = env
        input_size = 1 + env.R * env.R + env.R * (env.tau_d + env.L) # time, passenger state, car state
        out_put_size = env.R * env.R
        self.linear_rellu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 250),
            nn.ReLU(),
            nn.Linear(250, 1),
            nn.ReLU()
        )#policy network
        self.dataset_size = 2000
        self.dataset = []#it contains various car states

    def forward(self, x):
        return self.linear_rellu_stack(x)
    def generateSamples(self):
        for i in range(self.dataset_size):
            data_single_trial = []
            state = self.env.reset()
            while self.env.city_time < self.env.time_horizon:
                feasible_act = False
                data_piece = []
                while not feasible_act and self.env.city_time < self.env.time_horizon:
                    action = self.env.action_space.sample()
                    state, action, reward, feasible_act = self.env.step(action)
                    if feasible_act == True:
                        data_piece.append(self.env.city_time)
                        data_piece.append(self.env.i)
                        data_piece.append(self.env.It)
                        data_piece.append(reward)
                        data_piece.append(action)
                        data_piece.append(state)
                data_single_trial.append(data_piece)
            self.dataset.append(data_single_trial)



