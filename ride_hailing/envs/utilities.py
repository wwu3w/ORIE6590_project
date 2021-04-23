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
    for i in perm:
        if i % batch_size == 0:
            if len(y_seg) > 0:
                X_batch.append(X_seg)
                y_batch.append(y_seg)
                X_seg = []
                y_seg = []
        else:
            X_seg.append(X[i])
            y_seg.append(y[i])
    return X_batch, y_batch


def trainValueNet(X, y, batch_size, model, loss_fn, optimizer):
    print("Training valueNet...")
    X_batch, y_batch = segmentTrainingData(X, y, batch_size)
    size_batch = len(y_batch)
    perm = np.random.permutation(size_batch)
    print("X:")
    print(len(X))
    print("X_batch")
    print(len(X_batch))
    train_iter = 300
    for j in range(train_iter):
        loss_sum = 0
        for i in range(size_batch):
            X_seg = X_batch[i]
            y_seg = y_batch[i]
            y_len = len(y_seg)
            X_seg = torch.Tensor(X_seg)
            y_seg = torch.Tensor(y_seg)
            y_seg = torch.reshape(y_seg, (y_len, 1))
            pred = model(X_seg)
            loss = loss_fn(pred, y_seg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()   
        print(f"loss: {loss_sum/size_batch:>7f}")







