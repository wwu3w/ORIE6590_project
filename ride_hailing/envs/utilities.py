import torch
import numpy as np

def ids_2dto1d(i, j, M, N):
    '''
    convert (i,j) in a M by N matrix to index in M*N list. (row wise)
    matrix: [[1,2,3], [4, 5, 6]]
    list: [0, 1, 2, 3, 4, 5, 6]
    index start from 0
    '''
    assert 0 <= i < M and 0 <= j < N
    index = i * N + j
    return index


def ids_1dto2d(ids, M, N):
    ''' inverse of ids_2dto1d(i, j, M, N)
        index start from 0
    '''
    i = ids / N
    j = ids - N * i
    return (i, j)


def get_neighbor_index(i, j):
    """
                 1
             6       2
                center
             5       3
                 4
    return index of neighbor 1, 2, 3, 4, 5,6 in the matrix
    """
    neighbor_matrix_ids = []
    if j % 2 == 0:
        neighbor_matrix_ids = [[i - 1, j    ],
                               [i,     j + 1],
                               [i + 1, j + 1],
                               [i + 1, j    ],
                               [i + 1, j - 1],
                               [i    , j - 1]]
    elif j % 2 == 1:
        neighbor_matrix_ids = [[i - 1, j    ],
                               [i - 1, j + 1],
                               [i    , j + 1],
                               [i + 1, j    ],
                               [i    , j - 1],
                               [i - 1, j - 1]]

    return neighbor_matrix_ids

def get_neighbor_list(i, j, M, N):
    neighbor_list = []
    for ii, jj in get_neighbor_index(i ,j):
        if ii in range(M) and jj in range(N):
            neighbor_list.append(ids_2dto1d(ii, jj, M, N))

    return neighbor_list

def segmentTrainingData(X, y, batch_size):
    datalength = len(y)
    X_batch = []
    y_batch = []
    X_seg = []
    y_seg = []
    perm = np.random.permutation(datalength)
    counter = 0
    for i in perm:
        counter += 1
        if counter % batch_size == 0:
            if len(y_seg) > 0:
                X_batch.append(X_seg)
                y_batch.append(y_seg)
                X_seg = []
                y_seg = []
        else:
            X_seg.append(X[i])
            y_seg.append(y[i])
    return X_batch, y_batch

def segmentPolicyTrainingData(X, R, Act, Prob, batch_size):
    datalength = len(R)
    X_batch = []
    R_batch = []
    Act_batch = []
    Prob_batch = []
    X_seg = []
    R_seg = []
    Act_seg = []
    Prob_seg = []
    perm = np.random.permutation(datalength)
    counter = 0
    for i in perm:
        counter += 1
        if counter % batch_size == 0:
            if len(R_seg) > 0:
                X_batch.append(X_seg)
                R_batch.append(R_seg)
                Act_batch.append(Act_seg)
                Prob_batch.append(Prob_seg)
                X_seg = []
                R_seg = []
                Act_seg = []
                Prob_seg = []
        else:
            X_seg.append(X[i])
            R_seg.append(R[i])
            Act_seg.append(Act[i])
            Prob_seg.append(Prob[i])
    return X_batch, R_batch, Act_batch, Prob_batch

def trainValueNet(X, y, batch_size, model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training valueNet...")
    X_batch, y_batch = segmentTrainingData(X, y, batch_size)
    size_batch = len(y_batch)
    train_iter = 8
    for j in range(train_iter):
        loss_sum = 0
        weight_sum = 0
        for i in range(size_batch):
            X_seg = X_batch[i]
            y_seg = y_batch[i]
            y_len = len(y_seg)
            X_seg = torch.Tensor(X_seg)
            y_seg = torch.Tensor(y_seg) / model.scale #scale down 10 times to improve training performance
            y_seg = torch.reshape(y_seg, (y_len, 1))
            pred = model(X_seg.to(device))
            loss = loss_fn(pred, y_seg.to(device))
            optimizer.zero_grad()
            lambda2 = 0.01
            all_linear2_params = torch.cat([x.view(-1) for x in model.parameters()])
            l2_regularization = lambda2 * torch.norm(all_linear2_params, 2)
            weight_sum += l2_regularization.item()
            loss = loss + l2_regularization
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Value Estimation loss: {loss_sum/size_batch:>7f}")
    print("ValueNet Training Completed.")

def trainPolicyNet(X, R, Act, Prob, policymodel, batch_size, optimizer,  valuefnc):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training policyNet...")
    X_batch, R_batch, Act_batch, Prob_batch = segmentPolicyTrainingData(X, R, Act, Prob, batch_size)
    size_batch = len(R_batch)
    train_iter = 1
    for j in range(train_iter):
        loss_sum = 0
        for i in range(size_batch):
            X_seg = X_batch[i]
            X_seg = torch.Tensor(X_seg)
            R_seg = R_batch[i]
            Act_seg = Act_batch[i]
            Prob_seg = Prob_batch[i]
            pred = []
            for x in X_seg:
                pred.append(policymodel(x.to(device)))
            pred = torch.stack(pred)
            R_seg = torch.Tensor(R_seg)
            Act_seg = torch.Tensor(Act_seg)
            Prob_seg = torch.Tensor(Prob_seg)
            loss = policymodel.evalCost(X_seg.to(device), pred.to(device), R_seg.to(device), Act_seg.to(device), Prob_seg.to(device), valuefnc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"PolicyTrain Objective: {-loss_sum / size_batch:>7f}")
    print("policyNet Training Completed.")







