import numpy as np
import matplotlib.tri as tri
from CartPole import *

# task 1.2 changes of state: model Y = X(T) - X(0)

"""initialisation"""
cartpole1 = CartPole()

# scans of variables

"""setting the state scans"""
x_scan = np.linspace(-5,5,100)
x_dot_scan = np.linspace(-10,10,100)
theta_scan = np.linspace(np.pi-0.5,np.pi,100)
theta_dot_scan = np.linspace(10,15,100)

# """Y plot for x_dot scan"""
# Y0 = []
# for x in x_scan:
#     X = np.array([x,0,np.pi,0])
#     cartpole1.setState(X)
#     cartpole1.performAction()
#     Y = np.array(cartpole1.getState())
#     Y0.append((Y-X))
# Y0 = np.array(Y0)

# """Y plot for x_dot scan"""
# Y1 = []
# for x_dot in x_dot_scan:
#     X = np.array([0,x_dot,np.pi,0])
#     cartpole1.setState(X)
#     cartpole1.performAction()
#     Y = np.array(cartpole1.getState())
#     Y1.append((Y-X))
# Y1 = np.array(Y1)

# """Y plot for theta scan"""
# Y2 = []
# for theta in theta_scan:
#     X = np.array([0,0,theta,0])
#     cartpole1.setState(X)
#     cartpole1.performAction()
#     Y = np.array(cartpole1.getState())
#     Y2.append((Y-X))
# Y2 = np.array(Y2)

# """Y plot for theta_dot scan"""
# Y3 = []
# for theta_dot in theta_dot_scan:
#     X = np.array([0,0,np.pi,theta_dot])
#     cartpole1.setState(X)
#     cartpole1.performAction()
#     Y = np.array(cartpole1.getState())
#     Y3.append((Y-X))
# Y3 = np.array(Y3)

# """change the arguments to plot different dimensions of Y"""
# scan_name = 'x'
# t_data = x_scan
# y_data = Y0

# """plotting"""

# fig, axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
# axs[0,0].plot(x_scan,Y0[:,0],label='change in location')
# axs[0,0].plot(x_scan,Y0[:,1],label='change in velocity')
# axs[0,0].plot(x_scan,Y0[:,2],label='change in pole angle')
# axs[0,0].plot(x_scan,Y0[:,3],label='change in pole velocity')
# axs[0,0].set_title('cart_location')
# axs[0,0].set_xlabel('x (m)')
# axs[0,0].set_ylabel('change in state Y')
# axs[0,1].plot(x_dot_scan,Y1[:,0],label='change in location')
# axs[0,1].plot(x_dot_scan,Y1[:,1],label='change in velocity')
# axs[0,1].plot(x_dot_scan,Y1[:,2],label='change in pole angle')
# axs[0,1].plot(x_dot_scan,Y1[:,3],label='change in pole velocity')
# axs[0,1].set_title('cart_velocity')
# axs[0,1].set_xlabel('x_dot (m/s)')
# axs[0,1].set_ylabel('change in state Y')
# axs[1,0].plot(theta_scan,Y2[:,0],label='change in location')
# axs[1,0].plot(theta_scan,Y2[:,1],label='change in velocity')
# axs[1,0].plot(theta_scan,Y2[:,2],label='change in pole angle')
# axs[1,0].plot(theta_scan,Y2[:,3],label='change in pole velocity')
# axs[1,0].set_title('pole_angle') 
# axs[1,0].set_xlabel('theta (rad)')
# axs[1,0].set_ylabel('change in state Y')
# axs[1,1].plot(theta_dot_scan,Y3[:,0],label='change in location')
# axs[1,1].plot(theta_dot_scan,Y3[:,1],label='change in velocity')
# axs[1,1].plot(theta_dot_scan,Y3[:,2],label='change in pole angle')
# axs[1,1].plot(theta_dot_scan,Y3[:,3],label='change in pole velocity')
# axs[1,1].set_title('pole_velocity')
# axs[1,1].set_xlabel('theta_dot (rad/s)')
# axs[1,1].set_ylabel('change in state Y')
# fig.suptitle('Y as a function of scans of state variables', fontsize=16)
# handles, labels = axs[1,1].get_legend_handles_labels()
# fig.legend(handles, labels)
# # fig.tight_layout()
# plt.show()


"""setting the state scans for contours"""
x_scan = np.linspace(-10,10,50)
x_dot_scan = np.linspace(-10,10,50)
theta_scan = np.linspace(-np.pi,np.pi,50)
theta_dot_scan = np.linspace(-15,15,50)
# theta_scan = np.linspace(np.pi-0.5,np.pi,100)
# theta_dot_scan = np.linspace(10,15,100)

# contour plots 4C2 = 6

"""first generate Z for the contours"""

Y01 = [] # x & x_dot
for x_dot in x_dot_scan:
    Y0 = []
    for x in x_scan:
        X = np.array([x,x_dot,np.pi,0])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y0.append((Y-X))
    Y01.append(Y0)
Y01 = np.array(Y01)

Y02 = [] # x & theta
for theta in theta_scan:
    Y0 = []
    for x in x_scan:
        X = np.array([x,0,theta,0])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y0.append((Y-X))
    Y02.append(Y0)
Y02 = np.array(Y02)

Y03 = [] # x & theta_dot
for theta_dot in theta_dot_scan:
    Y0 = []
    for x in x_scan:
        X = np.array([x,0,np.pi,theta_dot])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y0.append((Y-X))
    Y03.append(Y0)
Y03 = np.array(Y03)

Y12 = [] # x_dot & theta
for theta in theta_scan:
    Y1 = []
    for x_dot in x_dot_scan:
        X = np.array([0,x_dot,theta,0])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y1.append((Y-X))
    Y12.append(Y1)
Y12 = np.array(Y12)

Y13 = [] # x_dot & theta_dot
for theta_dot in theta_dot_scan:
    Y1 = []
    for x_dot in x_dot_scan:
        X = np.array([0,x_dot,np.pi,theta_dot])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y1.append((Y-X))
    Y13.append(Y1)
Y13 = np.array(Y13)

Y23 = [] # theta & theta_dot
for theta_dot in theta_dot_scan:
    Y2 = []
    for theta in theta_scan:
        X = np.array([0,0,theta,theta_dot])
        cartpole1.setState(X)
        cartpole1.performAction()
        Y = np.array(cartpole1.getState())
        Y2.append((Y-X))
    Y23.append(Y2)
Y23 = np.array(Y23)



"""plot the contours"""
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
