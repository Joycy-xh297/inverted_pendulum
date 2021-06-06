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

# generating dataset
N = 1000 # NO of datapoints
M = 640 # NO of data locations for basis function
lam = 10**(-4) # variance of data noise
cartpole1 = CartPole()
X = []
Y = []
for i in range(N):
    x = random.uniform(-5,5)
    x_dot = random.uniform(-10,10)
    theta = random.uniform(-np.pi,np.pi)
    theta_dot = random.uniform(-15,15)
    act = random.uniform(-20,20)
    Xn = np.array([x,x_dot,theta,theta_dot,act])
    X.append(Xn)
    cartpole1.setState(Xn[:-1])
    cartpole1.performAction(action=Xn[-1])
    Xn_1 = np.array(cartpole1.getState())
    Y.append(Xn_1-Xn[:-1])
X = np.array(X)
Y = np.array(Y)

M_ind = random.sample(range(N),M)
XM = np.array([X[ind] for ind in M_ind])
sigma = [np.std(X[:,i]) for i in range(X.shape[1])]
K_NM = kernel(X,XM,sigma)
K_MM = kernel(XM,XM,sigma)
alpha = fit(K_NM,K_MM,lam,Y)

def rolloutL(p):
    max_t = 5
    state1 = np.array([0,0,0.1,0,0])
    state2 = np.array([0,0,0.15,0,0])
    init_state = state1
    cartpole = CartPole()
    steps = int(max_t/cartpole.delta_time) # 0.2s per step
    Xn = init_state[:-1]
    Xn_new = Xn
    L_model = 0
    for i in range(steps):
        Xn = Xn_new
        cartpole.setState(Xn[:4])
        action = np.dot(p,Xn)
        cartpole.performAction(action)
        cartpole.remap_angle()
        Xn_new = np.array(cartpole.getState())
        L_model+=cartpole.loss()
    return L_model

p = np.array([-4,10,22,6])
res = minimize(rolloutL,p,method='Nelder-Mead')
p1 = res.x

def rolloutL(p):
    max_t = 8
    state1 = np.array([0,0,0.2,0,0])
    state2 = np.array([0,0,0.15,0,0])
    init_state = state1
    cartpole = CartPole()
    steps = int(max_t/cartpole.delta_time) # 0.2s per step
    Xn = init_state[:-1]
    Xn_new = Xn
    L_model = 0
    for i in range(steps):
        Xn = Xn_new
        cartpole.setState(Xn[:4])
        action = np.dot(p,Xn)
        cartpole.performAction(action)
        cartpole.remap_angle()
        Xn_new = np.array(cartpole.getState())
        L_model+=cartpole.loss()
    return L_model

p = np.array([-4,10,20,6])
res = minimize(rolloutL,p,method='Nelder-Mead')
p2 = res.x

def rolloutL(p):
    max_t = 15
    state1 = np.array([0,0,0.3,0,0])
    state2 = np.array([0,0,0.15,0,0])
    init_state = state1
    cartpole = CartPole()
    steps = int(max_t/cartpole.delta_time) # 0.2s per step
    Xn = init_state[:-1]
    Xn_new = Xn
    L_model = 0
    for i in range(steps):
        Xn = Xn_new
        cartpole.setState(Xn[:4])
        action = np.dot(p,Xn)
        cartpole.performAction(action)
        cartpole.remap_angle()
        Xn_new = np.array(cartpole.getState())
        L_model+=cartpole.loss()
    return L_model

p = np.array([-4,10,19,5])
res = minimize(rolloutL,p,method='Nelder-Mead')
p3 = res.x

# for p = np.array([-4,10,22,6]) 0.1 time 5
# [ 0.76159935  1.19163129 16.33100508  2.49359169]
# 1.0499156754339123

# for p = np.array([-4,10,20,6]) 0.2 time 8
# [ 1.0457688   1.52085753 17.4291556   2.67365322]
# 1.7797756937552591

# for p = np.array([-4,10,19,5]) 0.3 time 15
# [ 0.94966785  0.84584117 17.14604703  2.43567508]
# 3.7835108616676205


def rolloutp(max_t,init_state,p):
    steps = int(max_t/cartpole1.delta_time) # 0.2s per step
    Xn = init_state[:-1]
    Xn_new = Xn
    cartpole = CartPole()
    X_cartpole = [Xn]
    L_model = 0
    for i in range(steps):
        Xn = Xn_new
        cartpole.setState(Xn[:4])
        action = np.dot(p,Xn)
        cartpole.performAction(action)
        cartpole.remap_angle()
        Xn_new = cartpole.getState()
        X_cartpole.append(np.array(Xn_new))
        L_model+=loss(Xn_new)
    X_cartpole = np.array(X_cartpole)
    return X_cartpole[:-1], L_model

max_t = 10
X_cartpole1, _ = rolloutp(max_t,np.array([0,0,0.1,0,0]),p3)
X_cartpole2, _ = rolloutp(max_t,np.array([0,0,0.4,0,0]),p2)
X_cartpole3, _ = rolloutp(max_t,np.array([0,0,0.7,0,0]),p3)
"""plot the time evolution using p"""
t = np.arange(0,max_t,cartpole1.delta_time)
fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axs[0,0].plot(t,X_cartpole1[:,0],label='initial angle 0.1')
axs[0,0].plot(t,X_cartpole2[:,0],label='initial angle 0.4')
axs[0,0].plot(t,X_cartpole3[:,0],label='initial angle 0.7')
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('time (s)')
axs[0,0].set_ylabel('location (m)')
axs[0,0].autoscale()

axs[0,1].plot(t,X_cartpole1[:,1],label='initial angle 0.1')
axs[0,1].plot(t,X_cartpole2[:,1],label='initial angle 0.4')
axs[0,1].plot(t,X_cartpole3[:,1],label='initial angle 0.7')
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('time (s)')
axs[0,1].set_ylabel('velocity (m/s)')
axs[0,1].autoscale()

axs[1,0].plot(t,X_cartpole1[:,2],label='inital angle 0.1')
axs[1,0].plot(t,X_cartpole2[:,2],label='inital angle 0.4')
axs[1,0].plot(t,X_cartpole3[:,2],label='inital angle 0.7')
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel('time (s)')
axs[1,0].set_ylabel('angle (rad)')
axs[1,0].autoscale()

axs[1,1].plot(t,X_cartpole1[:,3],label='initial angle 0.1')
axs[1,1].plot(t,X_cartpole2[:,3],label='initial angle 0.4')
axs[1,1].plot(t,X_cartpole3[:,3],label='initial angle 0.7')
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel('time (s)')
axs[1,1].set_ylabel('angular velocity (rad/s)')
axs[1,1].autoscale()

handles, labels = axs[1,1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center')
fig.suptitle('time evolution under the policy given small initial angle', fontsize=16)

plt.show()