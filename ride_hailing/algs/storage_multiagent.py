
import torch
import numpy as np

class RolloutBufferMultiAgent:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.actions = [ [] for _ in range(num_agents)]
        self.states = [ [] for _ in range(num_agents)]
        self.next_states = [ [] for _ in range(num_agents)]
        self.logprobs = [ [] for _ in range(num_agents)]
        self.rewards = [ [] for _ in range(num_agents)]
        self.time = [ [] for _ in range(num_agents)]
        self.value_targets = [ [] for _ in range(num_agents)]
        self.total_rewards = [ [] for _ in range(num_agents)]

        self.curr_rewards = [ [] for _ in range(num_agents)]
        self.curr_value_target = [ [] for _ in range(num_agents)]


    def save(self, action, state, logprob, reward, time, next_state, agent_id):
        self.actions[agent_id].append(action)
        self.states[agent_id].append(state)
        self.next_states[agent_id].append(next_state)
        self.logprobs[agent_id].append(logprob)
        self.curr_rewards[agent_id].append(reward)
        self.time[agent_id].append(time)

    def update(self, gamma, final_state):
        for agent_id in range(self.num_agents):
            self.next_states[agent_id].append(torch.Tensor(final_state[agent_id]))
            total_reward = 0
            for reward in reversed(self.curr_rewards[agent_id]):
                total_reward = reward + gamma * total_reward
                self.curr_value_target[agent_id].insert(0, total_reward)
            self.total_rewards[agent_id].append(total_reward)

        #self.actions = np.concatenate((self.actions, self.curr_actions))
            self.rewards[agent_id] += self.curr_rewards[agent_id]
        #self.states = np.concatenate((self.states, self.curr_states))
        #self.next_states = np.concatenate((self.next_states, self.curr_next_states))
        #self.logprobs = np.concatenate((self.logprobs, self.curr_logprobs))
        #self.time = np.concatenate((self.time, self.curr_time))
            self.value_targets[agent_id] = += self.curr_value_target[agent_id]

            self.curr_rewards[agent_id] = []
            self.curr_value_target[agent_id] = []

    def clear(self):
        self.actions = [[] for _ in range(self.num_agents)]
        self.states = [[] for _ in range(self.num_agents)]
        self.next_states = [[] for _ in range(self.num_agents)]
        self.logprobs = [[] for _ in range(self.num_agents)]
        self.rewards = [[] for _ in range(self.num_agents)]
        self.time = [[] for _ in range(self.num_agents)]
        self.value_targets = [[] for _ in range(self.num_agents)]
        self.total_rewards = [[] for _ in range(self.num_agents)]

        self.curr_rewards = [[] for _ in range(self.num_agents)]
        self.curr_value_target = [[] for _ in range(self.num_agents)]
