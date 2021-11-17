import numpy as np
from numpy import linalg
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def read_dataset(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
        x = np.zeros([len(lines), 123])
        y = np.zeros([len(lines)])
        for (i, line) in enumerate(lines):
            y[i] = 0.0 if line[0] == '-' else 1.0
            features = line.strip().split()[1:]
            for feature in features:
                index = int(feature.split(':')[0]) - 1
                assert 0 <= index < 123
                x[i, index] = 1.0
        return x.T, y


def predict(X, y, w):
    mu = 1.0 / (np.exp(-w.dot(X)) + 1.0) - 0.5
    labels = mu >= 0
    return np.sum(labels == y) / y.shape[0]


def IRLS(X, y, tX, ty, lam=0.001, threshold=0.00001, max_iter=20):
    train_acc, test_acc = [], []

    # X: [d, n], y: [n]
    d, n = X.shape

    # w: [d]
    w = np.ones(d)

    count = 0
    while True:
        # mu: [n], > 0
        mu = 1.0 / (np.exp(-w.dot(X)) + 1.0)

        # A, A_inv: [n, n]
        s = mu * (1 - mu) + 0.01
        A = sparse.diags(s)
        A_inv = sparse.diags(1 / s)

        # z:
        z = X.T.dot(w) - A_inv.dot(mu - y)

        # XAX^\intercal + \lambda I
        M = (A.dot(X.T)).T.dot(X.T) + lam * np.eye(d)
        M_inv = linalg.inv(M)

        # wt: [d]
        wt = A.dot(M_inv.dot(X).T).T.dot(z)

        # delta and update
        delta = np.sum(np.abs(w - wt)) / np.sum(np.abs(w))
        w = wt
        train_acc.append(predict(X, y, w))
        test_acc.append(predict(tX, ty, w))
        count += 1
        if delta < threshold or count == max_iter:
            break
    return w, train_acc, test_acc


# Read datasets
train_x, train_y = read_dataset('a9a/a9a')
test_x, test_y = read_dataset('a9a/a9a.t')

# Test different lambdas and draw
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xticks(range(1, 21))
ax2.set_xticks(range(1, 21))
for lam in [0, 0.1, 1, 10, 100, 1000]:
    print('Testing \\lambda={} ...'.format(lam))
    _, train_acc, test_acc = IRLS(train_x, train_y, test_x, test_y, lam=lam)
    print(train_acc, test_acc)
    int_axis = [i for i in range(1, len(train_acc) + 1)]
    ax1.plot(int_axis, train_acc, label='$\\lambda={}$'.format(lam))
    ax2.plot(int_axis, test_acc)
fig.legend()
fig.savefig('train.pdf')

