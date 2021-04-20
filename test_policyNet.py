from ride_hailing.envs.ride_hailing_env import *
from evaluate import *
from policyNetCentralized import *
from valueEstimator import *
from utilities import *
import torch
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

env = CityReal(R, tau_d, L, H, arrival_rate, trip_dest_prob, travel_time, c_state)
policyNet = PolicyNet(env)
valuefnc = valueEstimator(env)
epochs = 10
learning_rate = 1e-3
batch_size = 100
loss_fn = nn.MSELoss()#for value network training
optimizer = torch.optim.SGD(valuefnc.parameters(), lr = learning_rate)
for i in range(epochs):
	valuefnc.generateSamples(policyNet)
	X, y = valuefnc.oneReplicateEstimation()
	trainValueNet(X, y, batch_size, valuefnc, loss_fn, optimizer)


