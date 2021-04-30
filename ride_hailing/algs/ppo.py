import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class PPO:
    def __init__(self, gamma, K_epochs, buffer, model, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = buffer

        self.policy = model


        self.policy_old = model
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MSELoss = nn.MSELoss()

    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state)#.to(device)
            #print(state.shape[0])
            action, action_logprob = self.policy_old.act(state)

        return action.item(), state, action, action_logprob

    def batchsample(self, mini_batch_size):
        batch_size = len(self.buffer.rewards)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),mini_batch_size,drop_last=True)
        return sampler

    def update(self, mini_batch_size, lr_actor, lr_critic, eps_clip):
        self.eps_clip = eps_clip
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        #sampler = self.batchsample(mini_batch_size)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()  # .to(device)
        old_next_states = torch.squeeze(torch.stack(self.buffer.next_states, dim=0)).detach()  # .to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()  # .to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()  # .to(device)
        value_targets = torch.FloatTensor(self.buffer.value_targets)
        rewards = torch.FloatTensor(self.buffer.rewards)


        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            #for indices in sampler:



                # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            next_state_values = self.policy.critic(old_next_states)

                # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            next_state_values = torch.squeeze(next_state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
            advantages = rewards + next_state_values.detach() - state_values.detach()
            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-7)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 1 * self.MSELoss(state_values, value_targets) - 0.01 * dist_entropy

                # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

