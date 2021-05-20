from CartPole import *
import numpy as np
import matplotlib.pyplot as plt

"""task1.1 rollout simulation"""
cartpole1 = CartPole()
cartpole1.delta_time = 0.05
max_t = 5.0 # terminate time for the simulation
n = int(max_t/cartpole1.delta_time) # number of iterations for performAction
state1 = [0, 1, np.pi, 0] # nonzero cart velocity
state2 = [0, 0, np.pi, -14.7] # nonzero angular velocity
init_state = state1
cartpole1.setState(state=init_state)
# list of variables
x, x_dot, theta, theta_dot = [],[],[],[]
t = np.arange(0,max_t,cartpole1.delta_time)
# simulation
for i in range(n):
    cartpole1.performAction()
    # cartpole1.remap_angle()
    current_state = cartpole1.getState()
    x.append(current_state[0])
    x_dot.append(current_state[1])
    theta.append(current_state[2])
    theta_dot.append(current_state[3])
# plotting the results
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

"""task 1.2 change of state"""
cartpole1.reset()
x_scan = np.linspace(-5,5,100)
x_dot_scan = np.linspace(-10,10,100)
theta_scan = np.linspace(np.pi-0.5,np.pi,100)
theta_dot_scan = np.linspace(10,15,100)
# scans of variables
Y0, Y1, Y2, Y3 = [],[],[],[] # next state 
Y00,Y11,Y22,Y33 = [],[],[],[] # change in state
for x in x_scan:
    X = [x,0,np.pi,0]
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y0.append(Y)
    Y00.append(Y-X)
Y0 = np.array(Y0)
Y01 = np.array(Y00)
for x_dot in x_dot_scan:
    X = [0,x_dot,np.pi,0]
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y1.append(Y)
    Y11.append(Y-X)
Y1 = np.array(Y1)
Y11 = np.array(Y11)
for theta in theta_scan:
    X = [0,0,theta,0]
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y2.append(Y)
    Y22.append(Y-X)
Y2 = np.array(Y2)
Y21 = np.array(Y22)
for theta_dot in theta_dot_scan:
    X = [0,0,np.pi,theta_dot]
    cartpole1.setState(X)
    cartpole1.performAction()
    Y = np.array(cartpole1.getState())
    Y3.append(Y)
    Y33.append(Y-X)
Y3 = np.array(Y3)
Y31 = np.array(Y33)
# plotting scans of variables
# can change Y0, Y1, Y2, Y3 -> Y00,Y11,Y22,Y33 in the setting of Y = X(T)-X(0)
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
# generating contours
Y01,Y02,Y03,Y12,Y13,Y23 = [],[],[],[],[],[] 
for x_dot in x_dot_scan: # x & x_dot
    Y0 = []
    for x in x_scan:
        X = np.array([x,x_dot,np.pi,0])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y0.append((Y-X))
    Y01.append(Y0)
Y01 = np.array(Y01)
for theta in theta_scan: # x & theta
    Y0 = []
    for x in x_scan:
        X = np.array([x,0,theta,0])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y0.append((Y-X))
    Y02.append(Y0)
Y02 = np.array(Y02)
for theta_dot in theta_dot_scan: # x & theta_dot
    Y0 = []
    for x in x_scan:
        X = np.array([x,0,np.pi,theta_dot])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y0.append((Y-X))
    Y03.append(Y0)
Y03 = np.array(Y03)
for theta in theta_scan: # x_dot & theta
    Y1 = []
    for x_dot in x_dot_scan:
        X = np.array([0,x_dot,theta,0])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y1.append((Y-X))
    Y12.append(Y1)
Y12 = np.array(Y12) 
for theta_dot in theta_dot_scan: # x_dot & theta_dot
    Y1 = []
    for x_dot in x_dot_scan:
        X = np.array([0,x_dot,np.pi,theta_dot])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y1.append((Y-X))
    Y13.append(Y1)
Y13 = np.array(Y13)
for theta_dot in theta_dot_scan: # theta & theta_dot
    Y2 = []
    for theta in theta_scan:
        X = np.array([0,0,theta,theta_dot])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y2.append((Y-X))
    Y23.append(Y2)
Y23 = np.array(Y23)

def contour_plots(x,y,z,name):
    fig, axs = plt.subplots(2,2,figsize=(9,16),constrained_layout=True)
    x, y = np.meshgrid(x,y)
    triang = tri.Triangulation(x.flatten(), y.flatten())
    cntr1 = axs[0,0].tricontourf(triang, z[:,:,0].flatten())
    fig.colorbar(cntr1, ax=axs[0,0])
    axs[0,0].tricontour(triang,z[:,:,0].flatten())
    axs[0,0].set_xlabel(name[0])
    axs[0,0].set_ylabel(name[1])
    axs[0,0].set_title('change in cart_location')
    cntr2 = axs[0,1].tricontourf(triang, z[:,:,1].flatten())
    fig.colorbar(cntr2, ax=axs[0,1])
    axs[0,1].tricontour(triang,z[:,:,1].flatten())
    axs[0,1].set_xlabel(name[0])
    axs[0,1].set_ylabel(name[1])
    axs[0,1].set_title('change in cart_velocity')
    cntr3 = axs[1,0].tricontourf(triang, z[:,:,2].flatten())
    fig.colorbar(cntr3, ax=axs[1,0])
    axs[1,0].tricontour(triang,z[:,:,2].flatten())
    axs[1,0].set_xlabel(name[0])
    axs[1,0].set_ylabel(name[1])
    axs[1,0].set_title('change in pole angle')
    cntr4 = axs[1,1].tricontourf(triang, z[:,:,3].flatten())
    fig.colorbar(cntr4, ax=axs[1,1])
    axs[1,1].tricontour(triang,z[:,:,3].flatten())
    axs[1,1].set_xlabel(name[0])
    axs[1,1].set_ylabel(name[1])
    axs[1,1].set_title('change in pole_velocity')
    fig.suptitle('Y changes with {} and {}'.format(name[0],name[1]),fontsize=16)
    plt.show()

contour_plots(x_scan,x_dot_scan,Y01,['x','x_dot'])
contour_plots(x_scan,theta_scan,Y02,['x','theta'])
contour_plots(x_scan,theta_dot_scan,Y03,['x','theta_dot'])
contour_plots(x_dot_scan,theta_scan,Y12,['x_dot','theta'])
contour_plots(x_dot_scan,theta_dot_scan,Y13,['x_dot','theta_dot'])
contour_plots(theta_scan,theta_dot_scan,Y23,['theta', 'theta_dot'])

"""task 1.3 linear regression"""
def lin_reg(X,Y):
    X = np.matrix(X)
    XT = np.matrix.transpose(X)
    Y = np.matrix(Y)
    XT_X = np.matmul(XT, X)
    XT_Y = np.matmul(XT, Y)
    betas = np.matmul(np.linalg.inv(XT_X), XT_Y)
    return betas

cartpole1 = CartPole()
# generating dataset for training
X = []
Y = []
N = 500 # no of datapoints
for i in range(N):
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
# linear regression
coef = lin_reg(X,Y)
Y_predict = np.matmul(X,coef)
Y_predict = np.array(Y_predict)
# plotting scatter point contrsting real and predict data for next state
fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axs[0,0].scatter(Y[:,0]+X[:,0],Y_predict[:,0]+X[:,0])
axs[0,0].plot(Y[:,0]+X[:,0],Y[:,0]+X[:,0],color='r',alpha=0.7)
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('real (m)')
axs[0,0].set_ylabel('predict (m)')
axs[0,1].scatter(Y[:,1]+X[:,1],Y_predict[:,1]+X[:,1])
axs[0,1].plot(Y[:,1]+X[:,1],Y[:,1]+X[:,1],color='r',alpha=0.7)
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('real (m/s)')
axs[0,1].set_ylabel('predict (m/s)')
axs[1,0].scatter(Y[:,2]+X[:,2],Y_predict[:,2]+X[:,2])
axs[1,0].plot(Y[:,2]+X[:,2],Y[:,2]+X[:,2],color='r',alpha=0.7)
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel('real (rad)')
axs[1,0].set_ylabel('predict (rad)')
axs[1,1].scatter(Y[:,3]+X[:,3],Y_predict[:,3]+X[:,3])
axs[1,1].plot(Y[:,3]+X[:,3],Y[:,3]+X[:,3],color='r',alpha=0.7)
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel('real (rad/s)')
axs[1,1].set_ylabel('predict (rad/s)')
fig.suptitle('prediction vs real',fontsize=16)
plt.show()

# scans of variables can use the same segment of codes with a few changes in lines,
# so omitted here. the plot is generated using the following:
scan = 'x'
t = x_scan
y = Y0
yp = Y0_predict
fig, axs = plt.subplots(2,2,figsize=(9,16),constrained_layout=True)
axs[0,0].plot(t,y[:,0],label='true change')
axs[0,0].plot(t,yp[:,0],label='predicted change')
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel(scan)
axs[0,0].set_ylabel('location (m)')
axs[0,1].plot(t,y[:,1],label='true change')
axs[0,1].plot(t,yp[:,1],label='predicted change')
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel(scan)
axs[0,1].set_ylabel('velocity (m/s)')
axs[1,0].plot(t,y[:,2],label='true change')
axs[1,0].plot(t,yp[:,2],label='predicted change')
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel(scan)
axs[1,0].set_ylabel('angle (rad)')
axs[1,1].plot(t,y[:,3],label='true change')
axs[1,1].plot(t,yp[:,3],label='predicted change')
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel(scan)
axs[1,1].set_ylabel('angular velocity (rad/s)')
fig.suptitle('Y vs Y_predict {} scan'.format(scan), fontsize=16)
handles, labels = axs[1,1].get_legend_handles_labels()
fig.legend(handles, labels)
axs[0,0].autoscale()
axs[0,1].autoscale()
axs[1,0].autoscale()
axs[1,1].autoscale()
# fig.tight_layout()
plt.show()

"""task 1.4 simulating time evolution"""
Xn = np.array([0,0,np.pi,-14.7]) # change initial state here
max_t = 4
steps = int(max_t/cartpole1.delta_time) # 0.2s per step
# coef = np.matrix([[ 0.00078066,0.02962578,0.02914072,0.32273887],
#  [ 0.2013879,0.0440882,0.00710757,0.14244893],
#  [-0.01113819,-0.3663829,0.08273065,-0.08233264],
#  [ 0.01375745,0.11951334,0.20538774,0.03247904]]) # previous training result
# time evolution
X_cartpole = []
X_model = [] 
Xn1_new = Xn
Xn2_new = Xn
for i in range(steps):
    Xn1 = Xn1_new
    Xn2 = Xn2_new
    # Yn1 = model.predict([Xn1])[0]
    Yn1 = np.matmul(Xn1,coef)
    Yn1 = np.array(Yn1)
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
    # cartpole1.remap_angle()
    Xn2_new = np.array(cartpole1.getState())
    X_cartpole.append(Xn2_new)
X_cartpole = np.array(X_cartpole)
X_model = np.array(X_model)

# plotting the results
t = np.arange(0,max_t,cartpole1.delta_time)
fig, axs = plt.subplots(2,2,figsize=(9,16),constrained_layout=True)
axs[0,0].plot(t,X_cartpole[:,0],label='true dynamic')
axs[0,0].plot(t,X_model[:,0],label='model')
axs[0,0].set_title('cart_location')
axs[0,0].set_xlabel('time (s)')
axs[0,0].set_ylabel('location (m)')
axs[0,0].set_xlim([0,5])
axs[0,0].autoscale()

axs[0,1].plot(t,X_cartpole[:,1],label='true dynamics')
axs[0,1].plot(t,X_model[:,1],label='model')
axs[0,1].set_title('cart_velocity')
axs[0,1].set_xlabel('time (s)')
axs[0,1].set_ylabel('velocity (m/s)')
axs[0,1].set_xlim([0,5])
axs[0,1].autoscale()

axs[1,0].plot(t,X_cartpole[:,2],label='true dynamics')
axs[1,0].plot(t,X_model[:,2],label='model')
axs[1,0].set_title('pole_angle')
axs[1,0].set_xlabel('time (s)')
axs[1,0].set_ylabel('angle (rad)')
axs[1,0].set_xlim([0,5])
axs[1,0].autoscale()

axs[1,1].plot(t,X_cartpole[:,3],label='true dynamics')
axs[1,1].plot(t,X_model[:,3],label='model')
axs[1,1].set_title('pole_velocity')
axs[1,1].set_xlabel('time (s)')
axs[1,1].set_ylabel('angular velocity (rad/s)')
axs[1,1].set_xlim([0,5])
axs[1,1].autoscale()

handles, labels = axs[1,1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center')
fig.suptitle('model vs true dynamics given initial state: {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}'.format(
                                                                    Xn[0],Xn[1],Xn[2],Xn[3]), fontsize=16)

plt.show()