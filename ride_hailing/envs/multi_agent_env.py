import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ride_hailing.envs.utilities import *

RANDOM_SEED = 0  # unit test use this random seed.


class CityReal_ma(gym.Env):
    '''A real city is consists of R grids '''

    metadata = {'render.modes': ['human']}

    def __init__(self, M, N, R, tau_d, L, time_horizon, arrival_rate, trip_dest_prob, travel_time, c_state, capacity = 1000):
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
        self.M = M
        self.N = N
        self.R = R
        assert M * N == R
        self.tau_d = tau_d
        self.L = L
        self.curr_reward = 0
        self.total_reward = 0
        self.terminate = False
        self.num_neigbor = 6
        self.neighbor_list = np.zeros((self.R, self.num_neigbor)) - 1
        self.construct_map_neighbor()

        # parameters
        self.arrival_rate = arrival_rate
        self.trip_dest_prob = trip_dest_prob
        self.travel_time = travel_time
        self.time_horizon = time_horizon



        # States
        self.city_time = 0
        self.starting_c_state = np.asarray(c_state)
        self.c_state = self.starting_c_state  # car state R * tau_d
        self.p_state = np.zeros([self.L, self.R, self.R])  # passenger_state R * R
        self.It = 0
        self.i = 0
        self.patience_time = min(self.L + 1, self.time_horizon - self.city_time)


        # Action
        self.action_space = gym.spaces.Discrete(self.R + self.num_neigbor)
        space_dim = []
        for _ in range(self.time_horizon): #one-hot for time
            space_dim.append(int(1))
        for _ in range(self.R): #one-hot for grid
            space_dim.append(int(1))
        for _ in range(self.R * self.tau_d): #car_state
            space_dim.append(int(capacity))
        for _ in range(self.R): #local passenger_state
            space_dim.append(int(capacity))
        for _ in range(self.R - 1): #global passenger_state
            space_dim.append(int(capacity))
        self.observation_space = gym.spaces.MultiDiscrete(space_dim)


    def construct_map_neighbor(self):
        """Build node connection.
        """
        for grid_idx in range(self.R):
            i, j = ids_1dto2d(idx, self.M, self.N)

            for idx, neighbor_idx in enumerate(get_neighbor_list(i, j, self.M, self.N)):
                self.neighbor_list[grid_idx, idx] = neighbor_idx

    def generate_state(self, idx):
        state_time = np.zeros(self.time_horizon)
        state_time[int(self.city_time) - 1] = 1
        state_car = np.reshape(np.array(self.c_state), self.R * (self.tau_d+self.L))
        local_p = np.sum(self.p_state[:, idx, :], 0)
        global_p = np.sum(self.p_state, (0, 2))
        state_p_local = np.reshape(np.array(local_p), self.R)
        state_p_global = np.reshape(np.array(global_p), self.R)
        state_grid = np.zeros(self.R)
        state_grid[idx] = 1
        state = np.concatenate((state_time, state_car, state_p_local, state_p_global, state_grid), axis = None)
        return state



    def reset(self):
        self.total_reward = 0
        self.city_time = 0
        self.c_state = self.starting_c_state
        self.p_state = np.zeros(self.L, self.R, self.R)
        self.step_passenger_state_update()
        self.patience_time = min(self.L + 1, self.time_horizon - self.city_time)
        self.It = np.sum([self.c_state[_][0:self.patience_time] for _ in range(self.R)])
        self.i = 0
        self.terminate = False
        return self.generate_state()


    def step_passenger_state_update(self):
        for ll in range(self.L - 1):
            self.p_state[ll] = self.p_state[ll + 1]
        self.p_state[self.L - 1] = np.zeros([self.R, self.R])
        self.generate_trip_request()

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
                    self.p_state[self.L - 1, idx, dest_idx] += 1

    def step_car_state_update(self):
        for idx in range(self.R):
            for t in range(1, self.tau_d):
                n_car = self.c_state[idx][t]
                if n_car == 0:
                    continue
                self.c_state[idx][t - 1] += n_car
                self.c_state[idx][t] = 0

    def step_change_dest(self, dest1, dest2, tt1, tt2):
        self.c_state[dest1][tt1] -= 1
        self.c_state[int(dest2)][min(int(tt1 + tt2), self.tau_d-1)] += 1

    def step_time_update(self):
        self.i = 0
        self.step_car_state_update()
        self.step_passenger_state_update()
        self.city_time += 1
        self.patience_time = min(self.L + 1, self.time_horizon - self.city_time)
        self.It = np.sum([self.c_state[_][0:self.patience_time] for _ in range(self.R)])
        #print(self.i, self.It, self.city_time)

    def step(self, action):

        reward = 0


        #action = np.random.choice(range(self.R * self.R), 1, policy)[0]
        o, d = np.divmod(int(action), int(self.R))

        #ensure there exists available cars

        if np.sum(self.c_state[o,: self.patience_time]) <= 0:
            return self.generate_state(), action, reward, False

        for tt1 in range(self.L + 1):
            if self.c_state[o][tt1] > 0:
                break

        tt2 = self.travel_time[self.city_time + tt1][o][d]


        if np.sum(self.p_state[:,o,d]) > 0:
            reward = 1
            for ll in range(self.L):
                if self.p_state[ll,o,d] > 0:
                    self.p_state[ll,o,d] -= 1
                    break
        else:
            reward = 0
            if o == d:
                tt2 = 0

        self.step_change_dest(o, d, tt1, tt2)
        self.total_reward += reward

        self.i += 1

        output_state = self.generate_state()
        #print(self.i, self.It, self.city_time)
        while self.It <= self.i and self.city_time < self.time_horizon:
            self.step_time_update()
            if self.city_time == self.time_horizon:
                self.terminate = True






        return output_state, action, reward, True


