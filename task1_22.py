import numpy as np
from CartPole import *

# task 1.2 changes of state: model Y = X(T) - X(0)

"""initialisation"""
cartpole1 = CartPole()

"""setting the state scans"""
x_dot_scan = np.arange(-10,10,0.1)
theta_scan = np.arange(np.pi-0.1,np.pi,0.001)
theta_dot_scan = np.arange(14.5,15,0.01)

"""Y plot for x_dot scan"""
Y1 = []
for x_dot in x_dot_scan:
    X = np.array([0,x_dot,np.pi,0])
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y1.append((Y-X))
Y1 = np.array(Y1)
Y1_0 = Y1[:,0]
Y1_1 = Y1[:,1]
Y1_2 = Y1[:,2]
Y1_3 = Y1[:,3]

"""Y plot for theta scan"""
Y2 = []
for theta in theta_scan:
    X = np.array([0,0,theta,0])
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y2.append((Y-X))
Y2 = np.array(Y2)
Y2_0 = Y2[:,0]
Y2_1 = Y2[:,1]
Y2_2 = Y2[:,2]
Y2_3 = Y2[:,3]

"""Y plot for theta_dot scan"""
Y3 = []
for theta_dot in theta_dot_scan:
    X = np.array([0,0,np.pi,theta_dot])
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y3.append((Y-X))
Y3 = np.array(Y3)
Y3_0 = Y3[:,0]
Y3_1 = Y3[:,1]
Y3_2 = Y3[:,2]
Y3_3 = Y3[:,3]

"""change the arguments to plot different dimensions of Y"""
scan = 'x_dot'
t = x_dot_scan
y1 = Y1_0
y2 = Y1_1
y3 = Y1_2
y4 = Y1_3

"""plotting"""
fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axs[0,0].plot(t,y1)
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel(scan)
axs[0,0].set_ylabel('location (m)')
axs[0,1].plot(t,y2)
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel(scan)
axs[0,1].set_ylabel('velocity (m/s)')
axs[1,0].plot(t,y3)
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel(scan)
axs[1,0].set_ylabel('angle (rad)')
axs[1,1].plot(t,y4)
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel(scan)
axs[1,1].set_ylabel('angular velocity (rad/s)')
fig.suptitle('Y as a function of {} scan'.format(scan), fontsize=16)
# fig.tight_layout()
plt.show()