
from ride_hailing.envs.ride_hailing_env import *

def evaluate(model, env, numiters):
    r = 0
    r_square = 0
    # Start to iterate simulation loops
    for i in range(numiters):
        state = env.reset()
        # Loop for a whole time horizon of the ride-hailing system
        while env.city_time < env.time_horizon:
            feasible_act = False
            while not feasible_act and env.city_time < env.time_horizon:
                action = model(state, env)
                state, action, reward, feasible_act = env.step(action)
        #print(i, numiters)
        r += env.total_reward
        r_square += env.total_reward ** 2

    mean_reward = r/numiters
    sd_reward = np.sqrt(r_square/numiters - mean_reward ** 2)
    return mean_reward, sd_reward