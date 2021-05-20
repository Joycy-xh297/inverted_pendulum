import numpy as np
import numpy.random as random
# from sklearn import linear_model
from CartPole import *

# linear model Y=CX

def lin_reg(X,Y):
    X = np.matrix(X)
    XT = np.matrix.transpose(X)
    Y = np.matrix(Y)
    XT_X = np.matmul(XT, X)
    XT_Y = np.matmul(XT, Y)
    betas = np.matmul(np.linalg.inv(XT_X), XT_Y)
    return betas

cartpole1 = CartPole()

"""generating random states X and corresponding Y"""
X = []
Y = []
N = 500 # no of datapoints
for i in range(N):
    x = random.uniform([-5,5,1])[0]
    x_dot = random.uniform([-10,10,1])[0]
    theta = random.uniform([-np.pi,np.pi,1])[0]
    theta_dot = random.uniform([-15,15,1])[0]
    Xn = np.array([x,x_dot,theta,theta_dot])
    X.append(Xn)
    cartpole1.setState(Xn)
    cartpole1.performAction()
    Xn_1 = np.array(cartpole1.getState())
    Y.append(Xn_1-Xn)
X = np.array(X)
Y = np.array(Y)

"""performing linear regression"""
# model = linear_model.LinearRegression()
# model.fit(X,Y)
# coef = model.coef_
# print(coef)

# coef0 = lin_reg(X,Y[:,0])
# coef1 = lin_reg(X,Y[:,1])
# coef2 = lin_reg(X,Y[:,2])
# coef3 = lin_reg(X,Y[:,3])
# print(coef0)
# coef = np.column_stack([coef0,coef1,coef2,coef3])
coef = lin_reg(X,Y)
print(coef)
"""
matrix C is obtained as follows:
 [[ 9.68122344e-04  1.99391975e-01 -1.77179407e-02  1.00706026e-02]
 [-8.64559321e-03  1.87895679e-03 -5.04998587e-01  5.39049669e-02]
 [ 6.14853042e-04  1.62149669e-04  5.13113045e-02  1.90887788e-01]
 [-4.28778830e-02  9.90200361e-03 -5.31278489e-01 -1.71131758e-01]]
"""

"""generate model prediction"""
# Y_predict = model.predict(X)
Y_predict = np.matmul(X,coef)
Y_predict = np.array(Y_predict)
# print(Y_predict.shape)

"""plotting comparison - real & predict against current state"""
# fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)

# axs[0,0].scatter(X[:,0],Y[:,0],label='real')
# axs[0,0].scatter(X[:,0],Y_predict[:,0],label='predict')
# axs[0,0].set_title('cart_location')
# axs[0,0].set_xlabel('current location (m)')
# axs[0,0].set_ylabel('next location (m)')

# axs[0,1].scatter(X[:,1],Y[:,1],label='real')
# axs[0,1].scatter(X[:,1],Y_predict[:,1],label='predict')
# axs[0,1].set_title('cart_velocity')
# axs[0,1].set_xlabel('current velocity (m/s)')
# axs[0,1].set_ylabel('next velocity (m/s)')

# axs[1,0].scatter(X[:,2],Y[:,2],label='real')
# axs[1,0].scatter(X[:,2],Y_predict[:,2],label='predict')
# axs[1,0].set_title('pole_angle')
# axs[1,0].set_xlabel('current angle (rad)')
# axs[1,0].set_ylabel('next angle (rad)')

# axs[1,1].scatter(X[:,3],Y[:,3],label='real')
# axs[1,1].scatter(X[:,3],Y_predict[:,3],label='predict')
# axs[1,1].set_title('pole_velocity')
# axs[1,1].set_xlabel('current angular velocity (rad/s)')
# axs[1,1].set_ylabel('next angular velocity (rad/s)')

# fig.suptitle('prediction vs real',fontsize=16)
# handles, labels = axs[1,1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='middle center')
# plt.show()

"""plotting comparison - real vs predict"""

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

"""scans of variables"""
x_scan = np.linspace(-10,10,50)
x_dot_scan = np.linspace(-10,10,50)
theta_scan = np.linspace(-np.pi,np.pi,50)
theta_dot_scan = np.linspace(-15,15,50)


"""Y plot for x_dot scan"""
Y0 = []
Y0_predict = []
for x in x_scan:
    X = np.array([x,0,np.pi,0])
    y = np.matmul(X,coef)
    Y0_predict.append(y)
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y0.append((Y-X))
Y0 = np.array(Y0)
Y0_predict = np.array(Y0_predict)
Y0_predict = Y0_predict[:,0,:]

"""Y plot for x_dot scan"""
Y1 = []
Y1_predict = []
for x_dot in x_dot_scan:
    X = np.array([0,x_dot,np.pi,0])
    y = np.matmul(X,coef)
    Y1_predict.append(y)
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y1.append((Y-X))
Y1 = np.array(Y1)
Y1_predict = np.array(Y1_predict)
Y1_predict = Y1_predict[:,0,:]

"""Y plot for theta scan"""
Y2 = []
Y2_predict = []
for theta in theta_scan:
    X = np.array([0,0,theta,0])
    y = np.matmul(X,coef)
    Y2_predict.append(y)
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y2.append((Y-X))
Y2 = np.array(Y2)
Y2_predict = np.array(Y2_predict)
Y2_predict = Y2_predict[:,0,:]

"""Y plot for theta_dot scan"""
Y3 = []
Y3_predict = []
for theta_dot in theta_dot_scan:
    X = np.array([0,0,np.pi,theta_dot])
    y = np.matmul(X,coef)
    Y3_predict.append(y)
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y3.append((Y-X))
Y3 = np.array(Y3)
Y3_predict = np.array(Y3_predict)
Y3_predict = Y3_predict[:,0,:]

scan = 'x'
t = x_scan
y = Y0
yp = Y0_predict
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