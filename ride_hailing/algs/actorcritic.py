import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()



        self.embed = nn.Embedding(361, 6)


        # critic
        self.linear_stack = nn.Sequential(
            nn.Linear(state_dim + 5, 400),
            nn.Tanh(),
            nn.Linear(400, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
            nn.Tanh(),
            nn.Linear(5, 1)
        )

    def forward(self, input):
        if input.dim() == 1:
            input = torch.unsqueeze(input, dim=0)
        time = input[:,0].int()
        #print(input.size())
        time_embed = self.embed(time)
        input = torch.cat((time_embed, input[:,1:]), 1)
        output = self.linear_stack(input)
        return torch.squeeze(output)

    def evaluate_critic(self, state):

        return self.forward(state)



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()



        self.embed = nn.Embedding(360, 6)

        self.linear_stack = nn.Sequential(
            nn.Linear(state_dim + 5, 400),
            nn.Tanh(),
            nn.Linear(400, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
            nn.Tanh(),
            nn.Linear(5, action_dim),
            nn.Softmax(dim=0)
        )


    def forward(self, input):
        if input.dim() == 1:
            input = torch.unsqueeze(input, dim=0)
        time = input[:, 0].int()
        # print(input.size())
        time_embed = self.embed(time)
        input = torch.cat((time_embed, input[:, 1:]), 1)
        output = self.linear_stack(input)
        return torch.squeeze(output)

    def act(self, state, mask):

        action_probs = self.forward(state) * mask
        #print(mask)
        action_probs = action_probs / sum(action_probs)
        #print(action_probs)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()


    def evaluate_actor(self, state, action):

        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        #state_values = self.critic(state)

        return action_logprobs, action_probs, dist_entropy

    def evaluate_KL_divergence(self, state, old_probs):
        action_probs = self.forward(state)
        kl_div = F.kl_div(old_probs.log(), action_probs, None, None, 'sum').item()
        return kl_div



