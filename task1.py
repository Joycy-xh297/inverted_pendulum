from CartPole import *
import numpy as np
import matplotlib.pyplot as plt

# to simulate a rollout

# stable equilibrium position
# nonzero initial cart velocity or angular velocity
# no applied force

"""initialisation"""
cartpole1 = CartPole()
max_t = 200.0 # terminate time for the simulation
n = int(max_t/cartpole1.delta_time) # number of iterations for performAction

state1 = [0, 5, np.pi, 0] # nonzero cart velocity
state2 = [0, 0, np.pi, -0.1] # nonzero angular velocity
init_state = state2

x = []
x_dot = []
theta = []
theta_dot = []
t = np.arange(0,max_t,cartpole1.delta_time)

cartpole1.setState(state=init_state)

"""run the simulation"""
for i in range(n):
    cartpole1.performAction()
    cartpole1.remap_angle()
    current_state = cartpole1.getState()
    x.append(current_state[0])
    x_dot.append(current_state[1])
    theta.append(current_state[2])
    theta_dot.append(current_state[3])

    # print(cartpole1.getState())

"""plotting the results"""
fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axs[0,0].plot(t,x)
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('time (s)')
axs[0,0].set_ylabel('location (m)')
axs[0,1].plot(t,x_dot)
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('time (s)')
axs[0,1].set_ylabel('velocity (m/s)')
axs[1,0].plot(t,theta)
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel('time (s)')
axs[1,0].set_ylabel('angle (rad)')
axs[1,1].plot(t,theta_dot)
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel('time (s)')
axs[1,1].set_ylabel('angular velocity (rad/s)')
fig.suptitle('initial state: {}'.format(init_state), fontsize=16)
# fig.tight_layout()
plt.show()


