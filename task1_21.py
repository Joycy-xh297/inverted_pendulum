import numpy as np
from CartPole import *

# task 1.2 changes of state

"""initialisation"""
cartpole1 = CartPole()

"""setting the state scans"""
x_scan = np.linspace(-5,5,100)
x_dot_scan = np.linspace(-10,10,100)
theta_scan = np.linspace(np.pi-0.5,np.pi,100)
theta_dot_scan = np.linspace(10,15,100)

"""Y plot for x scan"""
Y0 = []
for x in x_scan:
    X = [x,0,np.pi,0]
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = cartpole1.getState()
    Y0.append(Y)
Y0 = np.array(Y0)

"""Y plot for x_dot scan"""
Y1 = []
for x_dot in x_dot_scan:
    X = [0,x_dot,np.pi,0]
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = cartpole1.getState()
    Y1.append(Y)
Y1 = np.array(Y1)

"""Y plot for theta scan"""
Y2 = []
for theta in theta_scan:
    X = [0,0,theta,0]
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = cartpole1.getState()
    Y2.append(Y)
Y2 = np.array(Y2)

"""Y plot for theta_dot scan"""
Y3 = []
for theta_dot in theta_dot_scan:
    X = [0,0,np.pi,theta_dot]
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = cartpole1.getState()
    Y3.append(Y)
Y3 = np.array(Y3)

"""change the arguments to plot different dimensions of Y"""
scan = 'x'
t = x_scan
y = Y0

"""plotting"""
# fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
# axs[0,0].plot(t,y[:,0])
# axs[0,0].set_title('cart_location')
# axs[0,0].set_xlabel(scan)
# axs[0,0].set_ylabel('location (m)')
# axs[0,1].plot(t,y[:,1])
# axs[0,1].set_title('cart_velocity')
# axs[0,1].set_xlabel(scan)
# axs[0,1].set_ylabel('velocity (m/s)')
# axs[1,0].plot(t,y[:,2])
# axs[1,0].set_title('pole_angle')
# axs[1,0].set_xlabel(scan)
# axs[1,0].set_ylabel('angle (rad)')
# axs[1,1].plot(t,y[:,3])
# axs[1,1].set_title('pole_velocity')
# axs[1,1].set_xlabel(scan)
# axs[1,1].set_ylabel('angular velocity (rad/s)')
# fig.suptitle('Y as a function of {} scan'.format(scan), fontsize=16)
# # fig.tight_layout()
# plt.show()

fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axs[0,0].plot(x_scan,Y0[:,0],label='next location')
axs[0,0].plot(x_scan,Y0[:,1],label='next velocity')
axs[0,0].plot(x_scan,Y0[:,2],label='next pole angle')
axs[0,0].plot(x_scan,Y0[:,3],label='next pole velocity')
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('x (m)')
axs[0,0].set_ylabel('next state Y')
axs[0,1].plot(x_dot_scan,Y1[:,0],label='next location')
axs[0,1].plot(x_dot_scan,Y1[:,1],label='next velocity')
axs[0,1].plot(x_dot_scan,Y1[:,2],label='next pole angle')
axs[0,1].plot(x_dot_scan,Y1[:,3],label='next pole velocity')
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('x_dot (m/s)')
axs[0,1].set_ylabel('next state Y')
axs[1,0].plot(theta_scan,Y2[:,0],label='next location')
axs[1,0].plot(theta_scan,Y2[:,1],label='next velocity')
axs[1,0].plot(theta_scan,Y2[:,2],label='next pole angle')
axs[1,0].plot(theta_scan,Y2[:,3],label='next pole velocity')
axs[1,0].set_title('pole_angle') 
axs[1,0].set_xlabel('theta (rad)')
axs[1,0].set_ylabel('next state Y')
axs[1,1].plot(theta_dot_scan,Y3[:,0],label='next location')
axs[1,1].plot(theta_dot_scan,Y3[:,1],label='next velocity')
axs[1,1].plot(theta_dot_scan,Y3[:,2],label='next pole angle')
axs[1,1].plot(theta_dot_scan,Y3[:,3],label='next pole velocity')
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel('theta_dot (rad/s)')
axs[1,1].set_ylabel('next state Y')
fig.suptitle('Y as a function of scans of state variables', fontsize=16)
handles, labels = axs[1,1].get_legend_handles_labels()
fig.legend(handles, labels)
# fig.tight_layout()
plt.show()