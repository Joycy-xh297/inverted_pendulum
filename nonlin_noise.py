from CartPole import *
import numpy as np
import random

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

        
N = 1000 # NO of datapoints
M = 800 # NO of data locations for basis function
lam = 10**(-4) # variance of data noise
cartpole1 = CartPole()

# added from 2.2
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
X_wn = X + np.random.normal(0,0.1,size=(N,5))
Y = np.array(Y)
Y_wn = Y + np.random.normal(0,0.1,size=(N,4))


M_ind = random.sample(range(N),M)
XM = np.array([X[ind] for ind in M_ind])
XM_wn = np.array([X_wn[ind] for ind in M_ind])
sigma = [np.std(X[:,i]) for i in range(X.shape[1])]
K_NM = kernel(X,XM,sigma)
K_MM = kernel(XM,XM,sigma)
K_NM_wn = kernel(X_wn,XM_wn,sigma)
K_MM_wn = kernel(XM_wn,XM_wn,sigma)
alpha = fit(K_NM,K_MM,lam,Y)
alpha_wn = fit(K_NM_wn,K_MM_wn,lam,Y_wn)
Y_predict = predict(X,XM,sigma,alpha)
Y_predict_wn = predict(X,XM_wn,sigma,alpha_wn)


"""time evolution using perfromAction and model"""
# setting parameters
max_t = 4.0
steps = int(max_t/cartpole1.delta_time) # 0.2s per step

Xn = np.array([0,0,np.pi,3,0.0])
X_cartpole = [Xn]
X_model = [Xn] 
X_model_wn = [Xn]

Xn1_new = Xn
Xn2_new = Xn
Xn3_new = Xn

for i in range(steps):
    Xn1 = Xn1_new
    Xn2 = Xn2_new
    Xn3 = Xn3_new
    Xn1 = Xn1.reshape(1,Xn1.shape[0])
    Yn1 = predict(Xn1,XM,sigma,alpha)
    Yn1.resize(Xn1.shape)
    Xn1_new = Xn1 + Yn1
    Xn1_new = np.array(Xn1_new[0])
    Xn1_new[2] = remap_angle(Xn1_new[2])
    X_model.append(Xn1_new)
    Xn3 = Xn3.reshape(1,Xn3.shape[0])
    Yn3 = predict(Xn3,XM_wn,sigma,alpha_wn)
    Yn3.resize(Xn3.shape)
    Xn3_new = Xn3 + Yn3
    Xn3_new = np.array(Xn3_new[0])
    Xn3_new[2] = remap_angle(Xn3_new[2])
    X_model_wn.append(Xn3_new)
    cartpole1.setState(Xn2[:-1])
    cartpole1.performAction()
    cartpole1.remap_angle()
    Xn2_new = cartpole1.getState()
    Xn2_new.resize(Xn2.shape)
    Xn2_new = np.array(Xn2_new)
    X_cartpole.append(Xn2_new)

X_cartpole = np.array(X_cartpole[:-1])
X_model = np.array(X_model[:-1])
X_model_wn = np.array(X_model_wn[:-1])

print('start plotting')
"""plotting"""
t = np.arange(0,max_t,cartpole1.delta_time)
fig, axs = plt.subplots(2,2,figsize=(9,16),constrained_layout=True)
print(t.shape)
print(X_cartpole.shape)
print(X_model.shape)
axs[0,0].plot(t,X_cartpole[:,0],label='true dynamic')
axs[0,0].plot(t,X_model[:,0],label='model')
axs[0,0].plot(t,X_model_wn[:,0],label='model with noise')
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('time (s)')
axs[0,0].set_ylabel('location (m)')
axs[0,0].set_xlim([0,5])
axs[0,0].autoscale()
# axs[0,0].set_ylim([-20,20])

axs[0,1].plot(t,X_cartpole[:,1],label='true dynamics')
axs[0,1].plot(t,X_model[:,1],label='model')
axs[0,1].plot(t,X_model_wn[:,1],label='model with noise')
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('time (s)')
axs[0,1].set_ylabel('velocity (m/s)')
axs[0,1].set_xlim([0,5])
axs[0,1].autoscale()
# axs[0,1].set_ylim([-20,20])

axs[1,0].plot(t,X_cartpole[:,2],label='true dynamics')
axs[1,0].plot(t,X_model[:,2],label='model')
axs[1,0].plot(t,X_model_wn[:,2],label='model with noise')
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel('time (s)')
axs[1,0].set_ylabel('angle (rad)')
axs[1,0].set_xlim([0,5])
axs[1,0].autoscale()
# axs[1,0].set_ylim([-20,20])

axs[1,1].plot(t,X_cartpole[:,3],label='true dynamics')
axs[1,1].plot(t,X_model[:,3],label='model')
axs[1,1].plot(t,X_model_wn[:,3],label='model with noise')
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel('time (s)')
axs[1,1].set_ylabel('angular velocity (rad/s)')
axs[1,1].set_xlim([0,5])
axs[1,1].autoscale()
# axs[1,1].set_ylim([-20,20])


handles, labels = axs[1,1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center')
fig.suptitle('model vs true dynamics given initial state: {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}'.format(
                                                                    Xn[0],Xn[1],Xn[2],Xn[3]), fontsize=16)


plt.show()
