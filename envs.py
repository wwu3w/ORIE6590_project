import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

RANDOM_SEED = 0  # unit test use this random seed.


class CityReal(gym.Env):
    '''A real city is consists of R grids '''

    metadata = {'render.modes': ['human']}

    def __init__(self, R, tau_d, L, time_horizon, arrival_rate, trip_dest_prob, travel_time, c_state, capacity = 1000):
        """
        :param mapped_matrix_int: 2D matrix: each position is either -100 or grid id from order in real data.
        :param order_num_dist: 144 [{node_id1: [mu, std]}, {node_id2: [mu, std]}, ..., {node_idn: [mu, std]}]
                            node_id1 is node the index in self.nodes
        :param idle_driver_dist_time: [[mu1, std1], [mu2, std2], ..., [mu144, std144]] mean and variance of idle drivers in
        the city at each time
        :param idle_driver_location_mat: 144 x num_valid_grids matrix.
        :param order_time_dist: [ 0.27380797,..., 0.00205766] The probs of order duration = 1 to 9
        :param order_price_dist: [[10.17, 3.34],   # mean and std of order's price, order durations = 10 minutes.
                                   [15.02, 6.90],  # mean and std of order's price, order durations = 20 minutes.
                                   ...,]
        :param onoff_driver_location_mat: 144 x 504 x 2: 144 total time steps, num_valid_grids = 504.
        mean and std of online driver number - offline driver number
        onoff_driver_location_mat[t] = [[-0.625       2.92350389]  <-- Corresponds to the grid in target_node_ids
                                        [ 0.09090909  1.46398452]
                                        [ 0.09090909  2.36596622]
                                        [-1.2         2.05588586]...]
        :param M:
        :param N:
        :param n_side:
        :param time_interval:
        :param l_max: The max-duration of an order
        :return:
        """
        # City.__init__(self, M, N, n_side, time_interval)
        self.R = R
        self.tau_d = tau_d
        self.L = L
        self.curr_reward = 0
        self.total_reward = 0



        # parameters
        self.arrival_rate = arrival_rate
        self.trip_dest_prob = trip_dest_prob
        self.travel_time = travel_time
        self.time_horizon = time_horizon



        # States
        self.city_time = 0
        self.starting_c_state = np.asarray(c_state)
        self.c_state = self.starting_c_state  # car state R * tau_d
        self.p_state = np.zeros([R, R])  # passenger_state R * R
        self.It = sum(self.c_state[:][:self.L + 1])
        self. i =0


        # Action
        self.action_space = gym.spaces.Discrete(self.R ** 2)
        space_dim = []
        space_dim.append(int(self.time_horizon))
        for _ in range(self.R * self.tau_d): #car_state
            space_dim.append(int(capacity))
        for _ in range(self.R * self.R): #passenger_state
            space_dim.append(int(capacity))
        self.observation_space = gym.spaces.MultiDiscrete(space_dim)



    def generate_state(self):
        state1 = np.reshape(np.array(self.c_state), self.R * self.tau_d)
        state2 = np.reshape(np.array(self.p_state), self.R ** 2)
        state = np.concatenate((state1, state2), axis = None)
        state = np.concatenate((np.array(self.city_time),state), axis = None)
        return state



    def reset(self):
        self.city_time
        self.c_state = self.starting_c_state
        self.p_state = np.zeros([self.R, self.R])
        self.It = sum(self.c_state[:][:self.L+1])
        self.i = 0
        return self.generate_state()


    def step_passenger_state_update(self):
        self.p_state = np.zeros([self.R, self.R])
        self.generate_trip_request()

    def generate_trip_request(self):
        for idx in range(self.R):
            lam = self.arrival_rate[self.city_time][idx]
            n_trip = np.random.poisson(lam, 1)[0]
            dest_prob = self.trip_dest_prob[self.city_time][idx]
            trip_dest = np.random.choice(range(self.R), n_trip, dest_prob)
            for dest_idx in trip_dest:
                self.p_state[idx][dest_idx] += 1

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
        self.c_state[dest2][tt1 + tt2] += 1

    def step(self, action):
        if self.i == 0:
            self.total_reward += self.curr_reward
            self.curr_reward = 0


        #action = np.random.choice(range(self.R * self.R), 1, policy)[0]
        o, d = np.divmod(int(action), int(self.R))

        #ensure there exists available cars
        assert sum(self.c_state[o][: L + 1]) > 0

        for tt1 in range(self.L + 1):
            if self.c_state[o][tt1] > 0:
                break

        tt2 = self.travel_time[self.city_time + tt1][o][d]
        self.step_change_dest(o, d, tt1, tt2)
        reward = 0
        if self.p_state[o][d] > 0:
            reward = 1
            self.p_state[o][d] -= 1
        self.curr_reward += reward

        self.i += 1

        if self.It <= self.i:
            self.city_time += 1
            self.step_car_state_update()
            self.step_passenger_state_update()
            self.It = sum(self.c_state[:][:self.L+1])




        return self.generate_state(), \
               action, reward, self.curr_reward


