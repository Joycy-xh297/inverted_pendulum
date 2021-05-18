import numpy as np
import numpy.random as random
from sklearn import linear_model
from CartPole import *

# linear model Y=CX, X(n+1)=X(n)+CX(n)

cartpole1 = CartPole()

"""generating random states X and corresponding Y"""
X = []
Y = []
for i in range(500):
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


"""setting time steps"""
max_t = 10
steps = int(max_t/cartpole1.delta_time) # 0.2s per step


# first initialise with a random state
x = random.uniform([-5,5,1])[0]
x_dot = random.uniform([-10,10,1])[0]
theta = random.uniform([-np.pi,np.pi,1])[0]
theta_dot = random.uniform([-15,15,1])[0]
Xn = np.array([x,x_dot,theta,theta_dot])
# or define initial conditions as below:
# to consider - oscillation around equilibrium, complete circular motion 
# Xn = np.array([0,0,np.pi,-13.5])

"""time evolution using perfromAction and model"""
X_cartpole = []
X_model = [] 

Xn1_new = Xn
Xn2_new = Xn

for i in range(steps):
    Xn1 = Xn1_new
    Xn2 = Xn2_new
    Yn1 = model.predict([Xn1])[0]
    Xn1_new = Xn1 + Yn1
    # # remapping the angle
    # Xn1_new[2] = remap_angle(Xn1_new[2])
    X_model.append(Xn1_new)
    cartpole1.setState(Xn2)
    cartpole1.performAction()
    Xn2_new = np.array(cartpole1.getState())
    X_cartpole.append(Xn2_new)

X_cartpole = np.array(X_cartpole)
X_model = np.array(X_model)

"""plotting"""
t = np.arange(0,max_t,cartpole1.delta_time)
fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axs[0,0].plot(t,X_cartpole[:,0],label='true dynamic')
axs[0,0].plot(t,X_model[:,0],label='model')
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('time (s)')
axs[0,0].set_ylabel('location (m)')
axs[0,1].plot(t,X_cartpole[:,1],label='true dynamics')
axs[0,1].plot(t,X_model[:,1],label='model')
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('time (s)')
axs[0,1].set_ylabel('velocity (m/s)')
axs[1,0].plot(t,X_cartpole[:,2],label='true dynamics')
axs[1,0].plot(t,X_model[:,2],label='model')
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel('time (s)')
axs[1,0].set_ylabel('angle (rad)')
axs[1,1].plot(t,X_cartpole[:,3],label='true dynamics')
axs[1,1].plot(t,X_model[:,3],label='model')
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel('time (s)')
axs[1,1].set_ylabel('angular velocity (rad/s)')

handles, labels = axs[1,1].get_legend_handles_labels()
fig.legend(handles, labels)
fig.suptitle('model vs true dynamics given initial state: {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}'.format(
                                                                    Xn[0],Xn[1],Xn[2],Xn[3]), fontsize=16)


plt.show()