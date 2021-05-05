from torch import nn
from ride_hailing.envs.utilities import *
from multiprocessing import Process, Queue, Array
from copy import deepcopy
device = "cuda" if torch.cuda.is_available() else "cpu"
class valueEstimator(nn.Module):
    def __init__(self, env):
        super(valueEstimator, self).__init__()
        self.env = env
        input_size = 1 + env.R * env.R + env.R * (env.tau_d + env.L) # time, passenger state, car state
        out_put_size = env.R * env.R
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU(),
        )#policy network
        self.dataset_size = 5
        self.dataset = []#it contains various car states
        self.scale = 10
        self.dataset_q = Queue()



    def forward(self, x):
        return self.linear_relu_stack(x)
    def generateData(self, policyNet):
        #Data lists
        S = []
        V = []
        R = []
        Action = []
        Prob = []
        processed_num = 0
        #process list
        procs = []
        print("Sampling Trajectories...")
        for i in range(self.dataset_size):
            p = Process(target=self.generateSamples, args=(policyNet,i))
            procs.append(p)
            p.start()
        while 1:
            if not self.dataset_q.empty():
                X_piece, y_piece, Action_piece, R_piece, Prob_piece = self.oneReplicateEstimation()
                S = S + X_piece
                V = V + y_piece
                R =  R + R_piece
                Action = Action + Action_piece
                Prob = Prob + Prob_piece
                processed_num += 1
            if processed_num >= self.dataset_size and self.dataset_q.empty():
                print("while loop out")
                break

        for p in procs:
            p.join()
        print("Sampling is done")
        return S, V, Action, R, Prob

    def generateSamples(self, policyNet, i):#generate data according to a policy net
        print("iter: " + str(i+1) + "/" +str(self.dataset_size))
        data_single_trial = []
        env = deepcopy(self.env)
        state = env.reset()
        state = torch.from_numpy(state.astype(np.float32))
        while env.city_time < env.time_horizon:
            data_piece = []
            action_distrib = policyNet(state)
            action = torch.multinomial(action_distrib/torch.sum(action_distrib), 1).item()
            action_prob = action_distrib[action]
            state_orig, action, reward, feasible_act = env.step(action)
            state = torch.from_numpy(state_orig.astype(np.float32))
            falseCount = 0
            if feasible_act == True:
                data_piece.append(reward)
                data_piece.append(state_orig)
                data_piece.append(action)
                data_piece.append(action_prob.item())
                
            else:
                while not feasible_act and env.city_time < env.time_horizon:
                    action_distrib[int(action)] = 1e-6
                    action = torch.multinomial(action_distrib/torch.sum(action_distrib), 1).item()
                    action_prob = action_distrib[action]
                    feasible_act = env.is_action_feasible(action)
                    falseCount += 1
                state_orig, action, reward, feasible_act = env.step(action)
                state = torch.from_numpy(state_orig.astype(np.float32))
                data_piece.append(reward)
                data_piece.append(state_orig)
                data_piece.append(action)
                data_piece.append(action_prob.item())
            data_single_trial.append(data_piece)
            if env.city_time%180 == 0 and env.i == 0:
                print("test envTime",env.city_time, "It", env.It, "total_reward", env.total_reward, "num_request", env.num_request, "FalseCount", falseCount)
        print("rate: " + str(i) + " : ", env.total_reward / env.num_request)
        self.dataset_q.put(data_single_trial)



    def oneReplicateEstimation(self):
        S = []
        V = []
        R = []
        Action = []
        Prob = []
        trial = self.dataset_q.get()
        #print(self.dataset_q.qsize())
        v_sum = 0
        for i in range(len(trial) - 1, -1, -1):
            datapiece = trial[i]
            r = datapiece[0]
            s = datapiece[1]
            a = datapiece[2]
            a_prob = datapiece[3]
            v_sum += r
            S.append(s)
            V.append(v_sum)
            Action.append(a)
            R.append(r)
            Prob.append(a_prob)
        return S, V, Action, R, Prob








