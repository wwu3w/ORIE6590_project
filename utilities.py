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
    for _ in range(train_iter):
        for i in perm:
            X_seg = X_batch[i]
            y_seg = y_batch[i]
            y_len = len(y_seg)
            X_seg = torch.Tensor(X_seg)
            y_seg = torch.reshape(torch.Tensor(y_seg), (y_len, 1))
            pred = model(X_seg)
            loss = loss_fn(pred, y_seg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                loss, current = loss.item(), i * len(X_seg)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(y):>5d}]")







