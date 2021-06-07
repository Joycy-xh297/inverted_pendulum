import numpy as np
from CartPole import *
import random
from sklearn.linear_model import LinearRegression

def mse(y1,y2):
    se = 0
    for i,j in zip(y1,y2):
#         se += abs((i-j)/i)
        # se += np.square(i-j)
        se += np.linalg.norm(i-j)**2
    return se/y1.shape[0]


def gen_train(N,s):
    """try generate the dataset and add noise together"""
    cartpole1 = CartPole()
    X = [] # true value for current state
    XX = [] # true value for next state
    X_wn = [] # observed current state
    Y = [] # true change in state
    Y_wn = [] # observed change in state
    for i in range(N):
        x = np.random.uniform([-5,5,1])[0]
        x_dot = np.random.uniform([-10,10,1])[0]
        theta = np.random.uniform([-np.pi,np.pi,1])[0]
        theta_dot = np.random.uniform([-15,15,1])[0]
        Xn = np.array([x,x_dot,theta,theta_dot])
        X.append(Xn)
        cartpole1.setState(Xn)
        cartpole1.performAction()
        cartpole1.remap_angle()
        Xn_1 = np.array(cartpole1.getState())
        XX.append(Xn_1)
        Y.append(Xn_1-Xn)

    X = np.array(X)
    Xnoise = np.random.normal(0,s,size=(N,4))
    X_wn = X + Xnoise
    XXnoise = np.random.normal(0,s,size=(N,4))
    XX_wn = XX + XXnoise
    Y = np.array(Y)
    Y_wn = np.array(XX_wn-X_wn)
    # coef = lin_reg(X,Y)
    # coef_wn = lin_reg(X_wn,Y_wn)
    reg1 = LinearRegression().fit(X,Y)
    reg2 = LinearRegression().fit(X_wn,Y_wn)
    inter1 = reg1.intercept_
    inter2 = reg2.intercept_
    coef = reg1.coef_
    coef_wn = reg2.coef_
    # print(coef)
    # print(coef_wn)
    Y_predict = np.matmul(X,coef)+inter1
    Y_predict = np.array(Y_predict)
    Y_predict_wn = np.matmul(X,coef_wn)+inter2
    Y_predict_wn = np.array(Y_predict_wn)
    return mse(Y_predict,Y), mse(Y_predict_wn,Y)


list_N = []
sigma_list = [0.05, 0.1, 0.2]
m_list = []
m_wnlist = []
N = 100 

for i in range(4):
    list_N.append(N)
    m = []
    m_wn = []
    for sig in sigma_list:
        a, b = gen_train(N,sig)
        m.append(a)
        m_wn.append(b)
    m_list.append(1.0*sum(m)/len(m))
    m_wnlist.append(m_wn)
    N *= 10
    
m_list = np.array(m_list)
m_wnlist = np.array(m_wnlist)
print(m_list)
print(m_wnlist)


"""plotting"""
plt.plot(list_N,m_list,label='without noise')
plt.plot(list_N,m_wnlist[:,0],label='observation noise sigma = 0.05')
plt.plot(list_N,m_wnlist[:,1],label='observation noise sigma = 0.1')
plt.plot(list_N,m_wnlist[:,2],label='observation noise sigma = 0.2')
plt.xscale('log')
plt.legend()
plt.show()

# fig, axs = plt.subplots(2,2,figsize=(9,16),constrained_layout=True)
# axs[0,0].plot(list_N,m[:,0],label='without noise')
# axs[0,0].plot(list_N,m_wn[:,0],label='with observation noise')
# axs[0,0].set_title('change in cart_location')
# axs[0,0].set_xlabel('N')
# axs[0,0].set_ylabel('mean percentage error')
# axs[0,0].set_xscale('log')
# axs[0,1].plot(list_N,m[:,1],label='without noise')
# axs[0,1].plot(list_N,m_wn[:,1],label='with observation noise')
# axs[0,1].set_title('change in cart_velocity')
# axs[0,1].set_xlabel('N')
# axs[0,1].set_ylabel('mean percentage error')
# axs[0,1].set_xscale('log')
# axs[1,0].plot(list_N,m[:,2],label='without noise')
# axs[1,0].plot(list_N,m_wn[:,2],label='with observation noise')
# axs[1,0].set_title('change in pole_angle')
# axs[1,0].set_xlabel('N')
# axs[1,0].set_ylabel('mean percentage error')
# axs[1,0].set_xscale('log')
# axs[1,1].plot(list_N,m[:,3],label='without noise')
# axs[1,1].plot(list_N,m_wn[:,3],label='with observation noise')
# axs[1,1].set_title('change in pole_velocity')
# axs[1,1].set_xlabel('N')
# axs[1,1].set_ylabel('mean percentage error')
# axs[1,1].set_xscale('log')
# fig.suptitle('prediction with/without observation noise', fontsize=16)
# handles, labels = axs[1,1].get_legend_handles_labels()
# fig.legend(handles, labels)
# # fig.tight_layout()
# plt.show()

