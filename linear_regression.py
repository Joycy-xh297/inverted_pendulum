import numpy as np

def lin_reg(X,Y):
    X = np.matrix(X)
    XT = np.matrix.transpose(X)
    Y = np.matrix(Y)
    XT_X = np.matmul(XT, X)
    XT_Y = np.matmul(XT, Y)
    betas = np.matmul(np.linalg.inv(XT_X), XT_Y)

    return betas