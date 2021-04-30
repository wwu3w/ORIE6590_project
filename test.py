from ride_hailing.envs.ride_hailing_env import *
from evaluate import *

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
proportion = lambda1/np.sum(lambda1)
for i in range(R):
	c_state[i,0]= int(N * proportion[i])
print(c_state)
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


state = env.reset()

def model(state, env):
    return env.action_space.sample()

numiters = 10  #iteration number
r= evaluate(model, env, numiters)
#print(np.mean(r) )
#print(np.sqrt(np.square(r).mean() - np.mean(r) ** 2))
print(r)
#print(r_sqr)