import gym
import numpy as np
from copy import deepcopy

RANDOM_SEED = 0  # unit test use this random seed.


class CityReal(gym.Env):
    '''A real city is consists of R grids '''

    metadata = {'render.modes': ['human']}

    def __init__(self, R, tau_d, L, time_horizon, arrival_rate, trip_dest_prob, travel_time, c_state, capacity = 1000):
        """
        :param R: interger, number of grids.
        :param tau_d: the longest trip travel time
        :param L: patience time
        :param time_horizon: time horizon of this system, number of time epochs
        :param arrival_rate: time_horizon * R: customer arrival rates [[1.8, 1.8, ...] ...]
        :param trip_dest_prob: time_horizon * R * R : probabilities among all destination grids [[[0.1, 0.2] ...] ...]
        :param travel_time: time_horizon * R * R : travel time for trip at time t from grid o to grid d [[[0.1, 0.2] ...] ...]

        """
        # City.__init__(self, M, N, n_side, time_interval)
        self.R = R
        self.tau_d = tau_d
        self.L = L
        self.curr_reward = 0
        self.total_reward = 0
        self.terminate = False
        self.num_request = 0
        self.car_assignment = []

        # parameters
        self.arrival_rate = arrival_rate
        self.trip_dest_prob = trip_dest_prob
        self.travel_time = travel_time
        self.time_horizon = time_horizon



        # States
        self.city_time = 0
        self.starting_c_state = np.asarray(c_state)
        self.c_state = deepcopy(self.starting_c_state)  # car state R * tau_d
        self.p_state = np.zeros([R, R])  # passenger_state R * R
        self.It = 0
        self.i = 0
        self.patience_time = min(self.L + 1, self.time_horizon - self.city_time)


        # Action dimension and State dimension
        #self.action_dim = self.R ** 2
        #self.state_dim = self.time_horizon + self.R * self.tau_d + self.R * self.R
        self.action_space = gym.spaces.Discrete(self.R ** 2)
        space_dim = []

        space_dim.append(int(self.time_horizon))
        for _ in range(self.R * (self.tau_d + self.L)): #car_state
            space_dim.append(int(capacity))
        for _ in range(self.R * self.R): #passenger_state
            space_dim.append(int(capacity))
        self.observation_space = gym.spaces.MultiDiscrete(space_dim)

        self.action_mask = np.ones(self.R ** 2)



    def generate_state(self):
        state_time = np.array([self.city_time])
        state_c = np.reshape(np.array(self.c_state), self.R * (self.tau_d+self.L))
        state_p = np.reshape(np.array(self.p_state), self.R ** 2)
        state = np.concatenate((state_time, state_c, state_p), axis=None)
        return state



    def reset(self):
        self.total_reward = 0
        self.num_request = 0
        self.city_time = 0
        self.c_state = deepcopy(self.starting_c_state) 
        self.step_passenger_state_update()
        self.patience_time = min(self.L, self.time_horizon - self.city_time)
        self.It = np.sum(self.c_state[: ,0:self.patience_time])
        self.i = 0
        self.terminate = False
        self.action_mask = np.ones(self.R ** 2)
        self.update_action_mask()
        self.car_assignment = []
        #print(self.starting_c_state)
        return self.generate_state(), self.action_mask


    def step_passenger_state_update(self):
        self.p_state = np.zeros([self.R, self.R])
        self.generate_trip_request()
        self.num_request += np.sum(self.p_state)

    def generate_trip_request(self):
        for idx in range(self.R):
            lam = self.arrival_rate[self.city_time][idx]
            n_trip = np.random.poisson(lam, 1)[0]
            dest_prob = self.trip_dest_prob[self.city_time][idx]
            #print(self.R,n_trip, dest_prob)
            if n_trip > 0:
                #print(dest_prob.shape())
                trip_dest = np.random.choice(self.R, n_trip, p = dest_prob)
                for dest_idx in trip_dest:
                    self.p_state[idx][dest_idx] += 1

    def step_car_state_update(self):
        while self.car_assignment:
            o, d, tt1, tt2 = self.car_assignment.pop()
            self.c_state[int(d), min(int(tt1 + tt2), self.tau_d + self.L - 1)] += 1
        self.car_assignment = []
        for idx in range(self.R):
            for t in range(1, self.tau_d):
                n_car = self.c_state[idx, t]
                #if n_car == 0:
                #    continue
                self.c_state[idx, t - 1] += n_car
                self.c_state[idx, t] = 0

    def step_change_dest(self, dest1, dest2, tt1, tt2):
        self.c_state[dest1][tt1] -= 1
        self.car_assignment.append([dest1, dest2, tt1, tt2])
        #self.c_state[int(dest2)][min(int(tt1 + tt2), self.tau_d+self.L-1)] += 1

    def step_time_update(self):
        self.i = 0
        self.step_car_state_update()
        self.step_passenger_state_update()
        self.city_time += 1
        self.patience_time = min(self.L, self.time_horizon - self.city_time)
        self.It = np.sum(self.c_state[:,0:self.patience_time])
        self.action_mask = np.ones(self.R ** 2)

        #print(self.i, self.It, self.city_time)

    def update_action_mask(self):
        self.action_mask = np.ones(self.R ** 2)
        grid_indices = np.where(np.sum(self.c_state[:, 0:self.patience_time], 1) <=0)[0]
        action_indices = []
        for idx in grid_indices:
            action_indices += range(idx * self.R, idx * self.R + self.R)
        self.action_mask[action_indices] = 0

    def step(self, action):

        reward = 0


        #action = np.random.choice(range(self.R * self.R), 1, policy)[0]
        o, d = np.divmod(int(action), int(self.R))

        #ensure there exists available cars
        #assert np.sum(self.c_state[o,: self.patience_time]) > 0

        if np.sum(self.c_state[o, 0: self.patience_time]) <= 0:
            return self.generate_state(), action, reward, [], False

        for tt1 in range(self.patience_time):
            if self.c_state[o,tt1] > 0:
                break

        tt2 = self.travel_time[self.city_time + tt1,o,d]


        if self.p_state[o][d] > 0:
            reward = 1
            self.p_state[o][d] -= 1
        else:
            reward = 0
            if o == d or tt1 > 0 :
                tt2 = 0

        self.step_change_dest(o, d, tt1, tt2)
        #print(self.city_time, reward, o, d)
        self.total_reward += reward

        self.i += 1
        #next_state = self.generate_state()

        #print(np.sum(self.c_state[o,: self.patience_time]))
        while np.sum(self.c_state[:, 0: self.patience_time]) <= 0 and not self.terminate:
            #print(self.c_state[:,: self.patience_time])
            self.step_time_update()
            #print(self.p_state)
            if self.city_time == self.time_horizon:
                #print(self.starting_c_state)
                self.terminate = True

        next_state = self.generate_state()
        self.update_action_mask()
        return next_state, action, reward, self.action_mask, True

    def is_action_feasible(self,action):
        o, d = np.divmod(int(action), int(self.R))
        if np.sum(self.c_state[o, 0: self.patience_time]) <= 0:
            return False
        else:
            return True


