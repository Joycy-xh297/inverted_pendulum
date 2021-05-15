import numpy as np
from CartPole import *

# task 1.2 changes of state

"""initialisation"""
cartpole1 = CartPole()

"""setting the state scans"""
x_dot_scan = np.arange(-10,10,0.1)
theta_scan = np.arange(-np.pi,np.pi,0.1)
theta_dot_scan = np.arange(-15,15,0.1)

"""Y plot for x_dot scan"""
Y1 = []
for x_dot in x_dot_scan:
    X = [0,x_dot,np.pi,0]
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = cartpole1.getState()
    Y1.append(Y)


"""Y plot for theta scan"""
Y2 = []
for theta in theta_scan:
    X = [0,0,theta,0]
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = cartpole1.getState()
    Y2.append(Y)

"""Y plot for theta_dot scan"""
Y3 = []
for theta_dot in theta_dot_scan:
    X = [0,0,np.pi,theta_dot]
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = cartpole1.getState()
    Y3.append(Y)



