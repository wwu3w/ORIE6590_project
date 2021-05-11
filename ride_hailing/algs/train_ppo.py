from ride_hailing.envs.ride_hailing_env import *
from ride_hailing.algs.ppo import *
from ride_hailing.algs.storage import *
from ride_hailing.algs.actorcritic import *
from datetime import datetime


# Parameters for initialization
R = 5
N = 1000 #number of cars
H = 360 #length of a working day
L = 5 #patience time
lambda1 = np.array([1.8,1.8,1.8,1.8,18])
lambda2 = np.array([12,8,8,8,2])
lambda3 = np.array([2,2,2,22,2])
P1 = np.array([[0.6, 0.1, 0.0, 0.3, 0.0],
	          [0.1, 0.6, 0.0, 0.3, 0.0],
	          [0.0, 0.0, 0.7, 0.3, 0.0],
	          [0.2, 0.2, 0.2, 0.2, 0.2],
	          [0.3, 0.3, 0.3, 0.1, 0.0]])
P2 = np.array([[0.1, 0.0, 0.0, 0.9, 0.0],
	          [0.0, 0.1, 0.0, 0.9, 0.0],
	          [0.0, 0.0, 0.1, 0.9, 0.0],
	          [0.05, 0.05, 0.05, 0.8, 0.05],
	          [0.0, 0.0, 0.0, 0.9, 0.1]])
P3 = np.array([[0.9, 0.05, 0, 0.05, 0],
	          [0.05, 0.9, 0, 0.05, 0],
	          [0, 0, 0.9, 0.1, 0],
	          [0.3, 0.3, 0.3, 0.05, 0.05],
	          [0, 0, 0, 0.1, 0.9]])
tau1 = np.array([[9, 15, 75, 12, 24],
	          	[15, 6, 66, 6, 18],
	          	[75, 66, 6, 60, 39],
	          	[15, 9, 60, 9, 15],
	          	[30, 24, 45, 15, 12]])
tau2 = np.array([[9, 15, 75, 12, 24],
	          	[15, 6, 66, 6, 18],
	          	[75, 66, 6, 60, 39],
	          	[12, 6, 60, 9, 15],
	          	[24, 18, 39, 15, 12]])
tau3 = np.array([[9, 15, 75, 12, 24],
	          	[15, 6, 66, 6, 18],
	          	[75, 66, 6, 60, 39],
	          	[12, 6, 60, 9, 15],
	          	[24, 18, 39, 15, 12]])
tau_d = np.max(tau1)
c_state = np.zeros((R, tau_d + L))
tot_unassigned = N
num_empty_cell = R*tau_d
init_dist = lambda1 / np.sum(lambda1)
for i in range(R):
	c_state[i,0] = int(N*init_dist[i])
#print(c_state)
travel_time = np.zeros((H,R,R))
trip_dest_prob = np.zeros((H,R,R))
arrival_rate = np.zeros((H,R))
for i in range(H):
	if i >= 0 and i <= 120:
		travel_time[i,:,:] = tau1
		trip_dest_prob[i,:,:] = P1
		arrival_rate[i,:] = lambda1
	elif i >= 121 and i <= 240:
		travel_time[i,:,:] = tau2
		trip_dest_prob[i,:,:] = P2
		arrival_rate[i,:] = lambda2
	else:
		travel_time[i,:,:] = tau3
		trip_dest_prob[i,:,:] = P3
		arrival_rate[i,:] = lambda3


# Initialize the whole environment

env = CityReal(R, tau_d, L, H, arrival_rate, trip_dest_prob, travel_time, c_state)
env.reset()


################ PPO hyperparameters ################

max_training_timesteps = 75  # break training loop if timeteps > max_training_timesteps

num_simulation = 5
K_epochs_value = 10  # update policy for K epochs in one PPO update
K_epochs_policy = 3
policy_early_stop = 0.012
eps_clip = 0.2  # clip parameter for PPO
gamma = 1  # discount factor

lr_actor = 0.00005  # learning rate for actor network
lr_critic = 0.0001  # learning rate for critic network

random_seed = 0  # set random seed if required (0 = no random seed)

#####################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


state_dim = env.observation_space.shape[0]
print(state_dim)
act_dim = env.action_space.n
Actor = Actor(state_dim, act_dim)
Critic = Critic(state_dim)
buffer = RolloutBuffer()
agent = PPO(gamma, K_epochs_policy, K_epochs_value, buffer, Actor, Critic, eps_clip, policy_early_stop, device)

start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

while time_step <= max_training_timesteps:

    state = env.reset()
    performance = []

    for _ in range(num_simulation):
        state, mask = env.reset()
        # Loop for a whole time horizon of the ride-hailing system
        while not env.terminate:
            #feasible_act = False
            #while not feasible_act and not env.terminate:
            # select action with policy
            action, state_buffer, action_buffer, logprob_buffer = agent.select_action(state, mask)
            state, action, reward, mask, feasible_act = env.step(action)

            agent.buffer.save(action_buffer, state_buffer, logprob_buffer, reward, env.city_time, torch.FloatTensor(state))
        agent.buffer.update(gamma)
        performance.append(env.total_reward/env.num_request)


    time_step += 1

    print("Timestep : {} \t\t Average Reward : {}".format(time_step, np.array(performance).mean()))

    # update PPO agent
    lr_c = max(1 - time_step/max_training_timesteps, 0.01) * lr_critic
    lr_a = max(1 - time_step/max_training_timesteps, 0.01) * lr_actor
    eps = max((1 - time_step/max_training_timesteps) * eps_clip, 0.01)
    agent.update(64, lr_a, lr_c, eps, device)

















