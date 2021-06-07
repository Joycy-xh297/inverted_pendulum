import numpy as np
from CartPole import *
import random
from sklearn.linear_model import LinearRegression

"""try generate the dataset and add noise together"""
cartpole1 = CartPole()
X = [] # true value for current state
XX = [] # true value for next state
X_wn = [] # observed current state
Y = [] # true change in state
Y_wn = [] # observed change in state
N = 1000 # no of datapoints
for i in range(N):
    x = np.random.uniform([-5,5,1])[0]
    x_dot = np.random.uniform([-10,10,1])[0]
    theta = np.random.uniform([-np.pi,np.pi,1])[0]
    theta_dot = np.random.uniform([-15,15,1])[0]
    Xn = np.array([x,x_dot,theta,theta_dot])
    X.append(Xn)
    cartpole1.setState(Xn)
    cartpole1.performAction()
    cartpole1.remap_angle()
    Xn_1 = np.array(cartpole1.getState())
    XX.append(Xn_1)
    Y.append(Xn_1-Xn)

X = np.array(X)
Xnoise = np.random.normal(0,0.1,size=(N,4))
X_wn = X + Xnoise
XXnoise = np.random.normal(0,0.1,size=(N,4))
XX_wn = XX + XXnoise
Y = np.array(Y)
Y_wn = np.array(XX_wn-X_wn)
# coef = lin_reg(X,Y)
# coef_wn = lin_reg(X_wn,Y_wn)
reg1 = LinearRegression().fit(X,Y)
reg2 = LinearRegression().fit(X_wn,Y_wn)
inter1 = reg1.intercept_
inter2 = reg2.intercept_
coef = reg1.coef_
coef_wn = reg2.coef_
print(coef)
print(coef_wn)
# Y_predict = np.matmul(X,coef)
# Y_predict = np.array(Y_predict)
# Y_predict_wn = np.matmul(X,coef_wn)
# Y_predict_wn = np.array(Y_predict_wn)

"""rollout of linear model"""
max_t = 4
Xn = np.array([0,0,np.pi,3])
steps = int(max_t/cartpole1.delta_time)
X_cartpole = [Xn]
X_model = [Xn]
X_model1 = [Xn]

Xn1_new = Xn
Xn2_new = Xn
Xn3_new = Xn

for i in range(steps):
    Xn1 = Xn1_new
    Xn2 = Xn2_new
    Xn3 = Xn3_new
    # Yn1 = model.predict([Xn1])[0]
    Yn1 = np.matmul(Xn1,coef)+inter1
    Yn1 = np.array(Yn1)
    Xn1_new = Xn1 + Yn1
    # remapping the angle
    # print(Xn1_new)
    Xn1_new = Xn1_new.flatten()
    Yn3 = np.matmul(Xn2,coef_wn)+inter2
    Yn3 = np.array(Yn3)
    Xn3_new = Xn3 + Yn3
    Xn3_new = Xn3_new.flatten()
    # print(Xn1_new)
    Xn1_new[2] = remap_angle(Xn1_new[2])
    Xn3_new[2] = remap_angle(Xn3_new[2])
    X_model.append(Xn1_new)
    X_model1.append(Xn3_new)
    cartpole1.setState(Xn2)
    cartpole1.performAction()
    # # remapping the angle for performAction
    cartpole1.remap_angle()
    Xn2_new = np.array(cartpole1.getState())
    X_cartpole.append(Xn2_new)

X_cartpole = np.array(X_cartpole[:-1])
X_model = np.array(X_model[:-1])
X_model1 = np.array(X_model1[:-1])

"""plotting"""
t = np.arange(0,max_t,cartpole1.delta_time)
fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axs[0,0].plot(t,X_cartpole[:,0],label='true dynamic')
axs[0,0].plot(t,X_model[:,0],label='model')
axs[0,0].plot(t,X_model1[:,0],label='model with noise')
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('time (s)')
axs[0,0].set_ylabel('location (m)')
axs[0,0].set_xlim([0,5])
axs[0,0].autoscale()
# axs[0,0].set_ylim([-20,20])

axs[0,1].plot(t,X_cartpole[:,1],label='true dynamics')
axs[0,1].plot(t,X_model[:,1],label='model')
axs[0,1].plot(t,X_model1[:,1],label='model with noise')
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('time (s)')
axs[0,1].set_ylabel('velocity (m/s)')
axs[0,1].set_xlim([0,5])
axs[0,1].autoscale()
# axs[0,1].set_ylim([-20,20])

axs[1,0].plot(t,X_cartpole[:,2],label='true dynamics')
axs[1,0].plot(t,X_model[:,2],label='model')
axs[1,0].plot(t,X_model1[:,2],label='model with noise')
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel('time (s)')
axs[1,0].set_ylabel('angle (rad)')
axs[1,0].set_xlim([0,5])
axs[1,0].autoscale()
# axs[1,0].set_ylim([-20,20])

axs[1,1].plot(t,X_cartpole[:,3],label='true dynamics')
axs[1,1].plot(t,X_model[:,3],label='model')
axs[1,1].plot(t,X_model1[:,3],label='model with noise')
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