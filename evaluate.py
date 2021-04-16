
import numpy as np

def evaluate(model, env, numiters):
    r = 0
    r_square = 0

    for _ in range(numiters):
        state = env.rest()
        while env.city_time < env.time_horizon:
            feasible_act = False
            # j = 0
            while not feasible_act and env.city_time < env.time_horizon:
                action = model(state, env)
                state, action, reward, cum_reward, feasible_act = env.step(action)
        r += env.total_reward
        r_square += env.total_reward ** 2
    return r/numiters, (r_square - r)/numiters