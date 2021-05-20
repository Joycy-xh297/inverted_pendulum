from CartPole import *
import numpy as np
import matplotlib.pyplot as plt

# task 1.1 simulate a rollout

"""initialisation"""
cartpole1 = CartPole()
cartpole1.delta_time = 0.05
max_t = 5.0 # terminate time for the simulation
n = int(max_t/cartpole1.delta_time) # number of iterations for performAction

"""set the initial state to start with"""
state1 = [0, 1, np.pi, 0] # nonzero cart velocity
state2 = [0, 0, np.pi, -14.7] # nonzero angular velocity
state3 = [0.71, 0.89, 0.53, -1.2]
init_state = state3
cartpole1.setState(state=init_state)

"""lists of system variables"""
x = []
x_dot = []
theta = []
theta_dot = []
t = np.arange(0,max_t,cartpole1.delta_time)

"""run the simulation"""
for i in range(n):
    cartpole1.performAction()
    # cartpole1.remap_angle()
    current_state = cartpole1.getState()
    x.append(current_state[0])
    x_dot.append(current_state[1])
    theta.append(current_state[2])
    theta_dot.append(current_state[3])

    # print(cartpole1.getState())

"""plotting the results"""
fig, axs = plt.subplots(2,2,figsize=(9,16),constrained_layout=True)
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
axs[1,0].set_xlim([0,5.0])
axs[1,1].plot(t,theta_dot)
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel('time (s)')
axs[1,1].set_ylabel('angular velocity (rad/s)')
fig.suptitle('initial state: {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}'.format(init_state[0],init_state[1],
                                                                    init_state[2],init_state[3]), fontsize=16)
# fig.tight_layout()
plt.show()


