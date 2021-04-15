import numpy as np
import gym
from envs import *
#test env
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
c_state = np.zeros((R, tau_d))
tot_unassigned = N
num_empty_cell = R*tau_d
for i in range(R):
	for j in range(tau_d):
		c_state[i,j] = int(np.floor(tot_unassigned/(num_empty_cell)))
		tot_unassigned -= c_state[i,j]
		num_empty_cell -= 1
travel_time = np.zeros((H,R,R))
for i in range(H):
	if i >= 0 and i <= 120:
		travel_time[i,:,:] = tau1
	elif i >= 121 and i <= 240:
		travel_time[i,:,:] = tau2
	else:
		travel_time[i,:,:] = tau3
env = CityReal(R, tau_d, L, H, lambda1, P1, travel_time, c_state)

for i in range(H):
	action = env.action_space.sample()
	state, action, reward, cum_reward = env.step(action)
	print(i)
