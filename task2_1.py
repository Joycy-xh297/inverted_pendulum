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

"""exploring the MSE as a function of increasing datapoints and basis functions"""
def mse(y1,y2):
    return np.linalg.norm(y1-y2)**2/y1.shape[0]

N = 500 # NO of datapoints
M = 320 # NO of data locations for basis function
lam = 10**(-3) # variance of data noise
cartpole1 = CartPole()

def gen_train(N,M):
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
        Xn_1 = np.array(cartpole1.getState())
        Y.append(Xn_1-Xn)
    X = np.array(X)
    Y = np.array(Y)
    # select randomly
    M_ind = random.sample(range(N),M)
    XM = np.array([X[ind] for ind in M_ind])
    sigma = [np.std(X[:,i]) for i in range(X.shape[1])]
    K_NM = kernel(X,XM,sigma)
    K_MM = kernel(XM,XM,sigma)
    # train and get prediction
    alpha = fit(K_NM,K_MM,lam,Y)
    Y_predict = predict(X,XM,sigma,alpha)
    # calculate mse
    return mse(Y_predict,Y)

# generate the dataset and train for the nonlinear model
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
    Xn_1 = np.array(cartpole1.getState())
    Y.append(Xn_1-Xn)
X = np.array(X)
Y = np.array(Y)
M_ind = random.sample(range(N),M)
XM = np.array([X[ind] for ind in M_ind])
sigma = [np.std(X[:,i]) for i in range(X.shape[1])]
K_NM = kernel(X,XM,sigma)
K_MM = kernel(XM,XM,sigma)
alpha = fit(K_NM,K_MM,lam,Y)

"""exploring the increase of N and M"""
mse_N = []
list_N = []
for i in range(6):
    mse_N.append(gen_train(N,M))
    list_N.append(N)
    N *= 2
mse_N = np.array(mse_N)
list_N = np.array(list_N)
# reset params
N = 1000
M = 10
mse_M = []
list_M = []
for i in range(6):
    mse_M.append(gen_train(N,M))
    list_M.append(M)
    M *= 2
mse_M = np.array(mse_M)
list_M = np.array(list_M)

"""plotting"""
fig, axs = plt.subplots(1,2,figsize=(20,8),constrained_layout=True)

axs[0].plot(list_N,mse_N)
axs[0].set_xlabel('data points N')
axs[0].set_ylabel('mse')
axs[0].set_xscale('log')
axs[0].autoscale()

axs[1].plot(list_M,mse_M)
axs[1].set_xlabel('basis functions M')
axs[1].set_ylabel('mse')
axs[1].set_xscale('log')
axs[1].autoscale()

fig.suptitle('mean squared error as a function of number of datapoints and basis functions',fontsize=16)
plt.show()

"""then plot 2D slices of target and the fit"""
"""scans of variables"""
x_scan = np.linspace(-10,10,50)
x_dot_scan = np.linspace(-10,10,50)
theta_scan = np.linspace(-np.pi,np.pi,50)
theta_dot_scan = np.linspace(-15,15,50)


"""Y plot for x scan"""
Y0 = []
Y0_predict = []
for x in x_scan:
    X = np.array([x,0,np.pi,0])
    cartpole1.setState(X)
    X = X.reshape(1,X.shape[0])
    y = predict(X,XM,sigma,alpha)
    Y0_predict.append(y)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y0.append((Y-X))
Y0 = np.array(Y0)
Y0_predict = np.array(Y0_predict)
Y0_predict = Y0_predict[:,0,:]
Y0 = Y0[:,0,:]

"""Y plot for x_dot scan"""
Y1 = []
Y1_predict = []
for x_dot in x_dot_scan:
    X = np.array([0,x_dot,np.pi,0])
    cartpole1.setState(X)
    X = X.reshape(1,X.shape[0])
    y = predict(X,XM,sigma,alpha)
    Y1_predict.append(y)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y1.append((Y-X))
Y1 = np.array(Y1)
Y1_predict = np.array(Y1_predict)
Y1_predict = Y1_predict[:,0,:]
Y1 = Y1[:,0,:]

"""Y plot for theta scan"""
Y2 = []
Y2_predict = []
for theta in theta_scan:
    X = np.array([0,0,theta,0])
    cartpole1.setState(X)
    X = X.reshape(1,X.shape[0])
    y = predict(X,XM,sigma,alpha)
    Y2_predict.append(y)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y2.append((Y-X))
Y2 = np.array(Y2)
Y2_predict = np.array(Y2_predict)
Y2_predict = Y2_predict[:,0,:]
Y2 = Y2[:,0,:]

"""Y plot for theta_dot scan"""
Y3 = []
Y3_predict = []
for theta_dot in theta_dot_scan:
    X = np.array([0,0,np.pi,theta_dot])
    cartpole1.setState(X)
    X = X.reshape(1,X.shape[0])
    y = predict(X,XM,sigma,alpha)
    Y3_predict.append(y)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y3.append((Y-X))
Y3 = np.array(Y3)
Y3_predict = np.array(Y3_predict)
Y3_predict = Y3_predict[:,0,:]
Y3 = Y3[:,0,:]

def scans(scan,t,y,yp):
    fig, axs = plt.subplots(2,2,figsize=(9,16),constrained_layout=True)
    axs[0,0].plot(t,y[:,0],label='true change')
    axs[0,0].plot(t,yp[:,0],label='predicted change')
    axs[0,0].set_title('cart_location')
    axs[0,0].set_xlabel(scan)
    axs[0,0].set_ylabel('location (m)')
    axs[0,1].plot(t,y[:,1],label='true change')
    axs[0,1].plot(t,yp[:,1],label='predicted change')
    axs[0,1].set_title('cart_velocity')
    axs[0,1].set_xlabel(scan)
    axs[0,1].set_ylabel('velocity (m/s)')
    axs[1,0].plot(t,y[:,2],label='true change')
    axs[1,0].plot(t,yp[:,2],label='predicted change')
    axs[1,0].set_title('pole_angle')
    axs[1,0].set_xlabel(scan)
    axs[1,0].set_ylabel('angle (rad)')
    axs[1,1].plot(t,y[:,3],label='true change')
    axs[1,1].plot(t,yp[:,3],label='predicted change')
    axs[1,1].set_title('pole_velocity')
    axs[1,1].set_xlabel(scan)
    axs[1,1].set_ylabel('angular velocity (rad/s)')
    fig.suptitle('Y vs Y_predict {} scan'.format(scan), fontsize=16)
    handles, labels = axs[1,1].get_legend_handles_labels()
    fig.legend(handles, labels)
    axs[0,0].autoscale()
    axs[0,1].autoscale()
    axs[1,0].autoscale()
    axs[1,1].autoscale()
    # fig.tight_layout()
    plt.show()

scans('x',x_scan,Y0,Y0_predict)
scans('x_dot',x_dot_scan,Y1,Y1_predict)
scans('theta',theta_scan,Y2,Y2_predict)
scans('theta_dot',theta_dot_scan,Y3,Y3_predict)