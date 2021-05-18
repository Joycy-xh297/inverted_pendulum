import numpy as np
import numpy.random as random
from sklearn import linear_model
from CartPole import *

# linear model Y=CX

cartpole1 = CartPole()

"""generating random states X and corresponding Y"""
X = []
Y = []
for i in range(5000):
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
model = linear_model.LinearRegression()
model.fit(X,Y)
# coef = model.coef_
# print(coef)
"""
matrix C is obtained as follows:
[[-0.00276529  0.19940025 -0.01613208  0.01138566]
 [-0.01184942  0.00290672 -0.47615685  0.0593577 ]
 [-0.00279054 -0.00095829  0.04580174  0.19259279]
 [-0.00458287  0.00717539 -0.55088481 -0.16938413]]
"""

"""generate model prediction"""
Y_predict = model.predict(X)

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

axs[0,0].scatter(Y[:,0],Y_predict[:,0])
axs[0,0].plot(Y[:,0],Y[:,0],color='r',alpha=0.7)
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('real (m)')
axs[0,0].set_ylabel('predict (m)')

axs[0,1].scatter(Y[:,1],Y_predict[:,1])
axs[0,1].plot(Y[:,1],Y[:,1],color='r',alpha=0.7)
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('real (m/s)')
axs[0,1].set_ylabel('predict (m/s)')

axs[1,0].scatter(Y[:,2],Y_predict[:,2])
axs[1,0].plot(Y[:,2],Y[:,2],color='r',alpha=0.7)
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel('real (rad)')
axs[1,0].set_ylabel('predict (rad)')

axs[1,1].scatter(Y[:,3],Y_predict[:,3])
axs[1,1].plot(Y[:,3],Y[:,3],color='r',alpha=0.7)
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel('real (rad/s)')
axs[1,1].set_ylabel('predict (rad/s)')

fig.suptitle('prediction vs real',fontsize=16)
plt.show()
