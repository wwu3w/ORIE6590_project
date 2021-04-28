

import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.time = []
        self.value_targets = []
        self.total_rewards = []

        self.curr_rewards = []
        self.curr_value_target = []


    def save(self, action, state, logprob, reward, time, next_state):
        self.actions.append(action)
        self.states.append(state)
        self.next_states.append(next_state)
        self.logprobs.append(logprob)
        self.curr_rewards.append(reward)
        self.time.append(time)

    def update(self, gamma):
        total_reward = 0
        for reward in reversed(self.curr_rewards):
            total_reward = reward + gamma * total_reward
            self.curr_value_target.insert(0, total_reward)
        self.total_rewards.append(total_reward)

        #self.actions = np.concatenate((self.actions, self.curr_actions))
        self.rewards = np.concatenate((self.rewards, self.curr_rewards))
        #self.states = np.concatenate((self.states, self.curr_states))
        #self.next_states = np.concatenate((self.next_states, self.curr_next_states))
        #self.logprobs = np.concatenate((self.logprobs, self.curr_logprobs))
        #self.time = np.concatenate((self.time, self.curr_time))
        self.value_targets = np.concatenate((self.value_targets, self.curr_value_target))

        self.curr_rewards = []
        self.curr_value_target = []

    def clear(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.time = []
        self.value_targets = []
        self.total_rewards = []

        self.curr_rewards = []
        self.curr_value_target = []
