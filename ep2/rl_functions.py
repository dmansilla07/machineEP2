import numpy as np
from util import randomize_in_place


def linear_regression_prediction(X, w):
    """
    Calculates the linear regression prediction.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :return: prediction
    :rtype: np.array(shape=(N, 1))
    """

    return X.dot(w)


def standardize(X):
    """
    Returns standardized version of the ndarray 'X'.

    :param X: input array
    :type X: np.ndarray(shape=(N, d))
    :return: standardized array
    :rtype: np.ndarray(shape=(N, d))
    """
    X_out = X.copy()
    X_out -= np.mean(X)
    X_out /= np.std(X)
    return X_out


def compute_cost(X, y, w):
    """
    Calculates  mean square error cost.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :return: cost
    :rtype: float
    """
    N = X.shape[0]
    J = 1.0/N*np.dot(np.transpose(np.matmul(X, w) - y), np.matmul(X, w) - y)
    return J[0][0]


def compute_wgrad(X, y, w):
    """
    Calculates gradient of J(w) with respect to w.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :return: gradient
    :rtype: np.array(shape=(d, 1))
    """
    yy = np.matmul(X, w)
    N = X.shape[0]
    M = X.shape[1]
    grad = np.ndarray(shape=(M, 1))
    for j in range(0, M):
        sum2 = (np.dot(np.transpose(X)[j], (yy-y)))*2/N
        grad[j] = sum2
    return grad


def batch_gradient_descent(X, y, w, learning_rate, num_iters):
    """
     Performs batch gradient descent optimization.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d, 1)), list, list
    """
    weights_history = [w.flatten()]
    
    cost_history = [compute_cost(X, y, w)]
    for i in range(0, num_iters):
        grad = compute_wgrad(X, y, w)
        w = w - learning_rate*grad
        weights_history.append(w.flatten())
        cost_history.append(compute_cost(X, y, w))

    return w, weights_history, cost_history 


def stochastic_gradient_descent(X, y, w, learning_rate, num_iters, batch_size):
    """
     Performs stochastic gradient descent optimization

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :param batch_size: size of the minibatch
    :type batch_size: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d, 1)), list, list
    """
    new_w = w.copy()
    N = X.shape[0]
    m = batch_size
    weights_history = [w.flatten()]
    cost_history = [compute_cost(X, y, w)]
    
    for i in range(0, num_iters):
        p = np.random.permutation(N)
        np.random.shuffle(p)
        new_x = np.ndarray(shape=(m, X.shape[1]))
        new_y = np.ndarray(shape=(m, y.shape[1]))
        for j in range(0, m):
            new_x[j] = X[p[j]]
            new_y[j] = y[p[j]]
        grad = compute_wgrad(new_x, new_y, new_w)
        new_w = new_w - learning_rate*grad
        weights_history.append(new_w.flatten())
        cost_history.append(compute_cost(X, y, new_w))    
    return np.asarray(new_w), weights_history, cost_history
