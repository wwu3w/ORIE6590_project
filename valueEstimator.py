import torch
import numpy as np
from torch import nn
from utilities import *
import gym
device = "cpu"
class valueEstimator(nn.Module):
    def __init__(self, env):
        super(valueEstimator, self).__init__()
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
        self.dataset_size = 1
        self.dataset = []#it contains various car states

    def forward(self, x):
        return self.linear_rellu_stack(x)
    def generateSamples(self, policyNet):#generate data according to a policy net
        for i in range(self.dataset_size):
            print("iter: " + str(i))
            data_single_trial = []
            state = self.env.reset()
            state = torch.from_numpy(state.astype(np.float32))
            init_action_distrib = policyNet(state)
            init_action = torch.multinomial(init_action_distrib,1).item()
            action = init_action
            while self.env.city_time < self.env.time_horizon:
                feasible_act = False
                data_piece = []
                while not feasible_act and self.env.city_time < self.env.time_horizon:
                    state, old_action, reward, feasible_act = self.env.step(action)
                    state = torch.from_numpy(state.astype(np.float32))
                    action_distrib = policyNet(state)
                    action = torch.multinomial(action_distrib, 1).item()
                    if feasible_act == True:
                        data_piece.append(reward)
                        data_piece.append(state)
                data_single_trial.append(data_piece)
            self.dataset.append(data_single_trial)
    def oneReplicateEstimation(self):
        S = []
        V = []
        for trial in self.dataset:
            v_sum = 0
            for i in range(len(trial) - 1, -1, 0):
                datapiece = trial[i]
                r = datapiece[0]
                s = datapiece[1].numpy()
                v_sum += r
                S.append(s)
                V.append(v_sum)
        return S, V








