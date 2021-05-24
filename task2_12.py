from CartPole import *
import numpy as np
from task2_11 import *


"""time evolution using perfromAction and model"""
# setting parameters
max_t = 4
steps = int(max_t/cartpole1.delta_time) # 0.2s per step
Xn = np.array([0,0,np.pi,3])

X_cartpole = [Xn]
X_model = [Xn] 

Xn1_new = Xn
Xn2_new = Xn

for i in range(steps):
    Xn1 = Xn1_new
    Xn2 = Xn2_new
    # get Xn1 kernel matrix for each of the variables
    aM1, aM2, aM3, aM4 = aM[0], aM[1], aM[2], aM[3]
    K1 = kernel(np.array([Xn1[0]]),XM[:,0],sig(X[:,0]))
    K2 = kernel(np.array([Xn1[1]]),XM[:,1],sig(X[:,1]))
    K3 = kernel(np.array([Xn1[2]]),XM[:,2],sig(X[:,2]))
    K4 = kernel(np.array([Xn1[3]]),XM[:,3],sig(X[:,3]))

    Yn10 = np.matmul(K1,aM1)
    Yn11 = np.matmul(K2,aM2)
    Yn12 = np.matmul(K3,aM3)
    Yn13 = np.matmul(K4,aM4)
    Yn1 = np.array([Yn10,Yn11,Yn12,Yn13])
    Xn1_new = Xn1 + Yn1
    # remapping the angle
    # print(Xn1_new)
    Xn1_new = Xn1_new.flatten()
    # print(Xn1_new)
    Xn1_new[2] = remap_angle(Xn1_new[2])
    X_model.append(Xn1_new)
    cartpole1.setState(Xn2)
    cartpole1.performAction()
    # # remapping the angle for performAction
    cartpole1.remap_angle()
    Xn2_new = np.array(cartpole1.getState())
    X_cartpole.append(Xn2_new)

X_cartpole = np.array(X_cartpole[:-1])
X_model = np.array(X_model[:-1])

print('start plotting')
"""plotting"""
t = np.arange(0,max_t,cartpole1.delta_time)
fig, axs = plt.subplots(2,2,figsize=(9,16),constrained_layout=True)
axs[0,0].plot(t,X_cartpole[:,0],label='true dynamic')
axs[0,0].plot(t,X_model[:,0],label='model')
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('time (s)')
axs[0,0].set_ylabel('location (m)')
axs[0,0].set_xlim([0,5])
axs[0,0].autoscale()
# axs[0,0].set_ylim([-20,20])

axs[0,1].plot(t,X_cartpole[:,1],label='true dynamics')
axs[0,1].plot(t,X_model[:,1],label='model')
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('time (s)')
axs[0,1].set_ylabel('velocity (m/s)')
axs[0,1].set_xlim([0,5])
axs[0,1].autoscale()
# axs[0,1].set_ylim([-20,20])

axs[1,0].plot(t,X_cartpole[:,2],label='true dynamics')
axs[1,0].plot(t,X_model[:,2],label='model')
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel('time (s)')
axs[1,0].set_ylabel('angle (rad)')
axs[1,0].set_xlim([0,5])
axs[1,0].autoscale()
# axs[1,0].set_ylim([-20,20])

axs[1,1].plot(t,X_cartpole[:,3],label='true dynamics')
axs[1,1].plot(t,X_model[:,3],label='model')
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