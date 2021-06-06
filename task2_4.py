from CartPole import *
import numpy as np
import random
from scipy.optimize import minimize

def kernel(X,Xi,sigma):
    K = np.zeros((X.shape[0],Xi.shape[0]))
    dim = X.shape[1]
    for i,x in enumerate(X):
        for j,xi in enumerate(Xi):
            sum = 0
            for k in range(dim):
                if k == 2:
                    sum += 1.0*np.sin((x[k]-xi[k])/2)**2/sigma[k]**2
                else:
                    sum += 1.0*(x[k]-xi[k])**2/sigma[k]**2
            K[i,j] = np.exp(-0.5*sum)
    return K

def fit(K_NM,K_MM,lam,Y):
    """return coefficients for each of the dimensions"""
    K_MN = np.transpose(K_NM)
    A = np.matmul(K_MN,K_NM) + lam * K_MM
    B = np.matmul(K_MN,Y)
    alpha = np.linalg.lstsq(A,B)[0]
    return alpha

def predict(X,XM,sigma,alpha):
    K_MN = kernel(X,XM,sigma)
    return np.matmul(K_MN,alpha)

def l(X,sigma):
    """X: state vector"""
    sum = 0
    for i,x in enumerate(X):
        sum += -0.5*np.linalg.norm(x)**2/sigma[i]**2
    return 1.0-np.exp(sum)

def modelL(p):
    max_t = 5
    state1 = np.array([0,0,0.1,0,0])
    state2 = np.array([0,0,0.3,0,0])
    init_state = state1
    cartpole = CartPole()
    steps = int(max_t/cartpole.delta_time) # 0.2s per step
    Xn = init_state
    Xn_new = Xn
    L_model = 0
    for i in range(steps):
        Xn = Xn_new
        # change the action term according to the policy
        Xn[-1] = np.dot(p,Xn[:-1])
        Xn = Xn.reshape(1,Xn.shape[0])
        Yn = predict(Xn,XM,sigma,alpha)
        Yn.resize(Xn.shape)
        Xn_new = Xn + Yn
        Xn_new = np.array(Xn_new[0])
        Xn_new[2] = remap_angle(Xn_new[2])
        L_model+=loss(Xn_new)
    return L_model

def rolloutp(max_t,init_state,p):
    steps = int(max_t/cartpole1.delta_time) # 0.2s per step
    Xn = init_state[:-1]
    Xn_new = Xn
    cartpole = CartPole()
    X_cartpole = [Xn]
    L_model = 0
    for i in range(steps):
        Xn = Xn_new
        # change the action term according to the policy
        cartpole.setState(Xn[:4])
        action = np.dot(p,Xn)
        cartpole.performAction(action)
        cartpole.remap_angle()
        Xn_new = cartpole.getState()
        X_cartpole.append(np.array(Xn_new))
        L_model+=loss(Xn_new)
    X_cartpole = np.array(X_cartpole)
    return X_cartpole[:-1], L_model

# N = 1000 # NO of datapoints
# M = 1000 # NO of data locations for basis function
# lam = 10**(-4) # variance of data noise
cartpole1 = CartPole()
# X = []
# Y = []
# for i in range(N):
#     x = random.uniform(-5,5)
#     x_dot = random.uniform(-10,10)
#     theta = random.uniform(-np.pi,np.pi)
#     theta_dot = random.uniform(-15,15)
#     act = random.uniform(-10,10)
#     Xn = np.array([x,x_dot,theta,theta_dot,act])
#     X.append(Xn)
#     cartpole1.setState(Xn[:-1])
#     cartpole1.performAction(action=Xn[-1])
#     Xn_1 = np.array(cartpole1.getState())
#     Y.append(Xn_1-Xn[:-1])
# X = np.array(X)
# Y = np.array(Y)

# M_ind = random.sample(range(N),M)
# XM = np.array([X[ind] for ind in M_ind])
# sigma = [np.std(X[:,i]) for i in range(X.shape[1])]
# K_NM = kernel(X,XM,sigma)
# K_MM = kernel(XM,XM,sigma)
# alpha = fit(K_NM,K_MM,lam,Y)

# p = np.array([0.76,1.19,16,2.49])
# res = minimize(modelL,p,method='Nelder-Mead')
# p = res.x
# final_loss = modelL(p)
# print(p)
# print(final_loss)

p = np.array([0.76763529, 1.5912036, 20.1163076, 2.58583496])
max_t = 10
X_cartpole1, _ = rolloutp(max_t,np.array([0,0,0.1,0,0]),p)
X_cartpole2, _ = rolloutp(max_t,np.array([0,0,0.3,0,0]),p)
X_cartpole3, _ = rolloutp(max_t,np.array([0,0,0.5,0,0]),p)
"""plot the time evolution using p"""
t = np.arange(0,max_t,cartpole1.delta_time)
fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axs[0,0].plot(t,X_cartpole1[:,0],label='initial angle 0.1')
axs[0,0].plot(t,X_cartpole2[:,0],label='initial angle 0.3')
axs[0,0].plot(t,X_cartpole3[:,0],label='initial angle 0.5')
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('time (s)')
axs[0,0].set_ylabel('location (m)')
axs[0,0].autoscale()

axs[0,1].plot(t,X_cartpole1[:,1],label='initial angle 0.1')
axs[0,1].plot(t,X_cartpole2[:,1],label='initial angle 0.3')
axs[0,1].plot(t,X_cartpole3[:,1],label='initial angle 0.5')
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('time (s)')
axs[0,1].set_ylabel('velocity (m/s)')
axs[0,1].autoscale()

axs[1,0].plot(t,X_cartpole1[:,2],label='initial angle 0.1')
axs[1,0].plot(t,X_cartpole2[:,2],label='initial angle 0.3')
axs[1,0].plot(t,X_cartpole3[:,2],label='initial angle 0.5')
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel('time (s)')
axs[1,0].set_ylabel('angle (rad)')
axs[1,0].autoscale()
# axs[1,0].set_ylim([-20,20])

axs[1,1].plot(t,X_cartpole1[:,3],label='initial angle 0.1')
axs[1,1].plot(t,X_cartpole2[:,3],label='initial angle 0.3')
axs[1,1].plot(t,X_cartpole3[:,3],label='initial angle 0.5')
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel('time (s)')
axs[1,1].set_ylabel('angular velocity (rad/s)')
axs[1,1].autoscale()

handles, labels = axs[1,1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center')
fig.suptitle('time evolution under the policy given small initial angle', fontsize=16)


plt.show()