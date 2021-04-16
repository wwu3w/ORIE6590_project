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
        self.It = 0
        self.i = 0
        self.patience_time = min(self.L + 1, self.time_horizon - self.city_time)


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
        state1 = np.reshape(np.array(self.c_state), self.R * (self.tau_d+self.L))
        state2 = np.reshape(np.array(self.p_state), self.R ** 2)
        state = np.concatenate((state1, state2), axis = None)
        state = np.concatenate((np.array(self.city_time),state), axis = None)
        return state



    def reset(self):
        self.total_reward = 0
        self.city_time = 0
        self.c_state = self.starting_c_state
        self.step_passenger_state_update()
        self.patience_time = min(self.L + 1, self.time_horizon - self.city_time)
        self.It = np.sum([self.c_state[_][0:self.patience_time] for _ in range(self.R)])
        self.i = 0
        self.terminate = False
        return self.generate_state()


    def step_passenger_state_update(self):
        self.p_state = np.zeros([self.R, self.R])
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

        if np.sum(self.c_state[o][: self.patience_time]) <= 0:
            return self.generate_state(), action, reward, False

        for tt1 in range(self.L + 1):
            if self.c_state[o][tt1] > 0:
                break

        tt2 = self.travel_time[self.city_time + tt1][o][d]


        if self.p_state[o][d] > 0:
            reward = 1
            self.p_state[o][d] -= 1
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


