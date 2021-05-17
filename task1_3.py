import numpy as np
import numpy.random as random
import sklearn as sk
from CartPole import *

# linear model Y=CX

cartpole1 = CartPole()

# generating random states X and corresponding Y
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

model = sk.linear_model.LinearRegression()
model.fit(X,Y)

