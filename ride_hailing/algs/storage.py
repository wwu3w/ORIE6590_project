


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def save(self, action, state, logprob, reward):
        self.actions.append(action)
        self.states.append(state)
        self.logprobs.append(logprob)
        self.rewards.append(reward)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
