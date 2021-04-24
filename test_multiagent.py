from ride_hailing.envs.multi_agent_env import *
from evaluate import *

# Parameters for initialization
R = 6
R_M = 2
R_N = 3
N = 1000 #number of cars
H = 360 #length of a working day
L = 5 #patience time
lambda1 = np.array([1.8,1.8,1.8,1.8,18,1.8])
lambda2 = np.array([12,8,8,8,2,5])
lambda3 = np.array([2,2,2,22,2,10])
P1 = np.array([[0.6, 0.1, 0.0, 0.3, 0.0, 0.0],
	          [0.1, 0.6, 0.0, 0.3, 0.0, 0.0],
	          [0.0, 0.0, 0.7, 0.2, 0.0, 0.1],
			  [0.0, 0.0, 0.7, 0.2, 0.0, 0.1],
	          [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
	          [0.3, 0.2, 0.3, 0.1, 0.0, 0.1]])
P2 = np.array([[0.1, 0.0, 0.0, 0.9, 0.0, 0.0],
	          [0.0, 0.1, 0.0, 0.9, 0.0, 0.0],
	          [0.0, 0.0, 0.1, 0.8, 0.0, 0.1 ],
	          [0.0, 0.0, 0.1, 0.8, 0.0, 0.1 ],
	          [0.05, 0.05, 0.05, 0.7, 0.05, 0.1],
	          [0.0, 0.0, 0.0, 0.9, 0.1, 0.0]])
P3 = np.array([[0.9, 0.05, 0, 0.05, 0, 0],
	          [0.05, 0.9, 0, 0.05, 0, 0],
	          [0, 0, 0.9, 0.1, 0, 0],
			   [0, 0, 0.9, 0.1, 0, 0],
	          [0.3, 0.3, 0.2, 0.05, 0.05, 0.1],
	          [0, 0, 0, 0.1, 0.8, 0.1]])
tau1 = np.array([[9, 15, 75, 12, 24,20],
	          	[15, 6, 66, 6, 18, 10],
	          	[75, 66, 6, 60, 39,10],
				 [75, 66, 6, 60, 39, 10],
	          	[15, 9, 60, 9, 15, 10],
	          	[30, 24, 45, 15, 12, 10]])
tau2 = np.array([[9, 15, 75, 12, 24, 13],
	          	[15, 6, 66, 6, 18, 12],
	          	[75, 66, 6, 60, 39, 34],
	          	[12, 6, 60, 9, 15,24],
				 [12, 6, 60, 9, 15, 24],
	          	[24, 18, 39, 15, 12, 6]])
tau3 = np.array([[9, 15, 75, 12, 24, 15],
	          	[15, 6, 66, 6, 18, 15],
				 [15, 6, 66, 6, 18, 15],
	          	[75, 66, 6, 60, 39, 15],
	          	[12, 6, 60, 9, 15, 15],
	          	[24, 18, 39, 15, 12, 15]])
tau_d = np.max(tau1)
c_state = np.zeros((R, tau_d + L))
tot_unassigned = N
num_empty_cell = R*tau_d
for i in range(R):
	for j in range(tau_d):
		c_state[i,j] = int(np.floor(tot_unassigned/(num_empty_cell)))
		tot_unassigned -= c_state[i,j]
		num_empty_cell -= 1
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

env = CityReal_ma(R_M, R_N, R, tau_d, L, H, arrival_rate, trip_dest_prob, travel_time, c_state)


state = env.reset()

def model(state, env):
    return env.action_space.sample()

numiters = 10  #iteration number
r, r_sqr = evaluate(model, env, numiters)
#print(np.mean(r) )
#print(np.sqrt(np.square(r).mean() - np.mean(r) ** 2))
print(r)
print(r_sqr)