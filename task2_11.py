import numpy as np
import random
from CartPole import *

# fit the model using nonlinear method
# target function being the change in state after one step

def kernel(X,Y,sigma):
    """for cart location, cart velocity and angular velocity"""
    K = np.zeros((X.shape[0],Y.shape[0]))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            K[i,j] = np.exp(-0.5*np.linalg.norm(x-y)**2/sigma)
    # XY = np.sum(np.multiply(X,Y),1)
    # K0 = XY + XY.T - 2 * X * Y.T
    # K = np.power(np.exp(-0.5/sigma**2), K0)
    return K

def sin_kernel(X,Y,sigma):
    """for pole angle"""
    K = np.zeros((X.shape[0],Y.shape[0]))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            K[i,j] = np.exp(-0.5*np.sin(np.linalg.norm(x-y)/2)**2/sigma)
    # XY = np.sum(np.multiply(X,Y),1)
    # K0 = np.sin(np.sqrt(XY + XY.T - 2 * X * Y.T)/2.0)**2
    # K = np.power(np.exp(-0.5/sigma**2), K0)
    return K

def sig(X):
    return np.std(X)

def fit(K_NM,K_MM,lam,Y):
    """return coefficients for each of the dimensions"""
    K_MN = np.transpose(K_NM)
    A = np.matmul(K_MN,K_NM) + lam * K_MM
    B = np.matmul(K_MN,Y)
    alpha = np.linalg.lstsq(A,B)[0]
    return alpha

N = 500 # NO of datapoints
M = 320 # NO of data locations for basis function
lam = 10**(-5) # variance of data noise
cartpole1 = CartPole()

# generate the dataset
X = []
Y = []
for i in range(N):
    x = random.uniform(-5,5)
    x_dot = random.uniform(-10,10)
    theta = random.uniform(-np.pi,np.pi)
    theta_dot = random.uniform(-15,15)
    Xn = np.array([x,x_dot,theta,theta_dot])
    X.append(Xn)
    cartpole1.setState(Xn)
    cartpole1.performAction()
    cartpole1.remap_angle()
    Xn_1 = np.array(cartpole1.getState())
    Y.append(Xn_1-Xn)
X = np.array(X)
Y = np.array(Y)
# Y2 = np.array([remap_angle(i) for i in Y[:,2]])
# print(Y2)
# Y[:,2] = Y2

# generate kernel matrices
# first select M places out
M_ind = random.sample(range(N),M)
XM = np.array([X[ind] for ind in M_ind])

# for K_NM
"""fitting and making prediction"""
aM = []
Y_predict = np.array([])
for i in range(4):
    if i == 2:
        k1 = sin_kernel(X[:,i],XM[:,i],sig(X[:,i]))
        k2 = sin_kernel(XM[:,i],XM[:,i],sig(X[:,i]))
    else:
        k1 = kernel(X[:,i],XM[:,i],sig(X[:,i]))
        k2 = kernel(XM[:,i],XM[:,i],sig(X[:,i]))
        # print(k1.shape,k2.shape)
    alp = fit(k1,k2,lam,Y[:,i])
    aM.append(np.array(alp))
    YN = np.matmul(k1,alp)
    # if i==2: # remap the angle
    #     YN = np.array([remap_angle(y) for y in YN])
    YN = YN.reshape(500,1)
    if i == 0:
        Y_predict = YN
    else:
        Y_predict = np.concatenate((Y_predict,YN),axis=1)
aM = np.array(aM)
Y_predict = np.array(Y_predict)
# print(al.shape)
# print(X.shape)
# print(Y.shape)
# print(Y_predict.shape)

"""scatter plot: Yp against Y"""
fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)

axs[0,0].scatter(Y[:,0]+X[:,0],Y_predict[:,0]+X[:,0])
axs[0,0].plot(Y[:,0]+X[:,0],Y[:,0]+X[:,0],color='r',alpha=0.7)
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('real (m)')
axs[0,0].set_ylabel('predict (m)')

axs[0,1].scatter(Y[:,1]+X[:,1],Y_predict[:,1]+X[:,1])
axs[0,1].plot(Y[:,1]+X[:,1],Y[:,1]+X[:,1],color='r',alpha=0.7)
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('real (m/s)')
axs[0,1].set_ylabel('predict (m/s)')

axs[1,0].scatter(Y[:,2]+X[:,2],Y_predict[:,2]+X[:,2])
axs[1,0].plot(Y[:,2]+X[:,2],Y[:,2]+X[:,2],color='r',alpha=0.7)
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel('real (rad)')
axs[1,0].set_ylabel('predict (rad)')

axs[1,1].scatter(Y[:,3]+X[:,3],Y_predict[:,3]+X[:,3])
axs[1,1].plot(Y[:,3]+X[:,3],Y[:,3]+X[:,3],color='r',alpha=0.7)
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel('real (rad/s)')
axs[1,1].set_ylabel('predict (rad/s)')

fig.suptitle('prediction vs real',fontsize=16)
plt.show()

