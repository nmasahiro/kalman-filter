import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def KalmanFilter(K, y, mu0, C0, F, G, H, Q, R):
    mu = mu0
    C = C0

    x_filter = np.zeros([K, 2])
    C_filter = np.zeros([K, 2, 2])

    x_filter[0] = mu.T
    C_filter[0] = C

    for k in range(1, K):
        # prediction:
        mu_ = F @ mu
        C_ = C @ F @ C.T + G @ Q @ G.T

        # filtering:
        S = H @ C_ @ H.T + R
        K = C_ @ H.T / S[0][0]
        #K = K.reshape(2, 1)

        mu = mu_ + K @ (y[k] - H @ mu_)
        C = (np.eye(2) - K @ H) @ C_

        x_filter[k] = mu.reshape(1, 2)
        C_filter[k] = C

    return x_filter, C_filter

def main():
    K = 30
    # initial distribution
    mu0 = np.zeros([2, 1])
    C0 = np.eye(2)

    # simulation parameter
    F = np.array([[0, -0.7], [1, -1.5]])
    G = np.array([[0.5], [1]])
    H = np.array([[0, 1.0]])

    # noise parameter
    # system noise variance
    Q = np.array([[1.0]])
    # observation noise variance
    R = np.array([[0.4]])

    # make true value and observation value
    true_x_log = np.zeros([K, 2])
    y = np.zeros([K, 1])
    true_x = np.array([[0.0], [0.0]])
    true_x_log[0] = true_x.reshape(1, 2)
    y[0] = H @ true_x + np.random.randn() * np.sqrt(R)
    for k in range(1, K):
        true_x = F @ true_x + G * np.random.randn() * np.sqrt(Q)
        true_x_log[k] = true_x.reshape(1, 2)
        y[k] = H @ true_x + np.random.randn() * np.sqrt(R)
    print("true_x:", true_x)
    print("y:", y)

    x_filter, C_filter = KalmanFilter(K, y, mu0, C0, F, G, H, Q, R)
    print("x_filter:", x_filter)
    print("C_filter:", C_filter)

    plt.figure()
    plt.plot(range(K), true_x_log[:, 0], label="true x1")
    plt.plot(range(K), x_filter[:, 0], label="estimated x1")
    
    plt.plot(range(K), true_x_log[:, 1], label="true x2")
    plt.plot(range(K), x_filter[:, 1], label="estimated x2")
    plt.legend()
    plt.savefig('x.pdf')



if __name__ == '__main__':
    main()
