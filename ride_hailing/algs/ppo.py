import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class PPO:
    def __init__(self, gamma, K_epochs_actor, K_epochs_critic, buffer, actor, critic, eps_clip, early_stop, device):
        self.early_stop = early_stop
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs_actor = K_epochs_actor
        self.K_epochs_critic = K_epochs_critic
        self.buffer = buffer

        self.actor = actor.to(device)
        self.critic = critic.to(device)

        # = model.to(device)
        #self.policy_old.load_state_dict(self.policy.state_dict())

        self.MSELoss = nn.MSELoss()

    def select_action(self, state, mask):

        with torch.no_grad():
            state = torch.FloatTensor(state)#.to(device)
            #print(state.shape[0])
            action, action_logprob = self.actor.act(state, mask)

        return action.item(), state, action, action_logprob

    def batchsample(self, mini_batch_size):
        batch_size = len(self.buffer.rewards)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),mini_batch_size,drop_last=True)
        return sampler

    def update(self, mini_batch_size, lr_actor, lr_critic, eps_clip, device):
        self.eps_clip = eps_clip
        self.optimizer_actor = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor}
        ])
        self.optimizer_critic = optim.Adam([
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])

        #sampler = self.batchsample(mini_batch_size)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_next_states = torch.squeeze(torch.stack(self.buffer.next_states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        value_targets = torch.FloatTensor(self.buffer.value_targets).to(device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(device)

        # Optimize value NN for K epochs
        for _ in range(self.K_epochs_critic):
            state_values = self.critic.evaluate_critic(old_states)
            state_values = torch.squeeze(state_values).to(device)
            loss = self.MSELoss(state_values, value_targets)

            self.optimizer_critic.zero_grad()
            loss.mean().backward()
            self.optimizer_critic.step()

        # Optimize policy NN for K epochs
        for _ in range(self.K_epochs_actor):

                # Evaluating old actions and values
            logprobs, old_action_probs, dist_entropy = self.actor.evaluate_actor(old_states, old_actions)
            dist_entropy.to(device)
            state_values = self.critic.evaluate_critic(old_states)#.to(device)
            next_state_values = self.critic.evaluate_critic(old_next_states)#.to(device)

                # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values).to(device)
            next_state_values = torch.squeeze(next_state_values).to(device)

                # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach()).to(device)

                # Finding Surrogate Loss
            advantages = rewards + next_state_values.detach() - state_values.detach()
            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-7).to(device)
            advantages.to(device)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy

                # take gradient step
            self.optimizer_actor.zero_grad()
            loss.mean().backward()
            self.optimizer_actor.step()

            # early stop criteria
            kl_div = self.actor.evaluate_KL_divergence(old_states, old_action_probs)
            if kl_div <= self.early_stop:
                break

        # Copy new weights into old policy
        #self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

