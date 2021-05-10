
from ride_hailing.envs.ride_hailing_env import *

def evaluate(model, env, numiters):
    r = []
    # Start to iterate simulation loops
    for i in range(numiters):
        state, _ = env.reset()
        # Loop for a whole time horizon of the ride-hailing system
        while env.city_time < env.time_horizon:
            feasible_act = False
            while not feasible_act and env.city_time < env.time_horizon:
                action = model(state, env)
                state, action, reward, _, feasible_act = env.step(action)
        print(env.total_reward, env.num_request)
        r.append(env.total_reward/env.num_request)
        #r_square += env.total_reward ** 2

    #mean_reward = r/numiters
    #sd_reward = np.sqrt(r_square/numiters - mean_reward ** 2)
    return r