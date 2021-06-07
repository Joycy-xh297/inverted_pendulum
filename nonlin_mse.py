from CartPole import *
import numpy as np
import random

def kernel(X,Xi,sigma):
    K = np.zeros((X.shape[0],Xi.shape[0]))
    dim = X.shape[1]
    for i,x in enumerate(X):
        for j,xi in enumerate(Xi):
            sum = 0
            for k in range(dim):
                if k == 2:
                    sum += 1.0*np.sin((x[k]-xi[k])/2)**2/sigma[k]**2
                else:
                    sum += 1.0*(x[k]-xi[k])**2/sigma[k]**2
            K[i,j] = np.exp(-0.5*sum)
    return K

def fit(K_NM,K_MM,lam,Y):
    """return coefficients for each of the dimensions"""
    K_MN = np.transpose(K_NM)
    A = np.matmul(K_MN,K_NM) + lam * K_MM
    B = np.matmul(K_MN,Y)
    alpha = np.linalg.lstsq(A,B)[0]
    return alpha

def predict(X,XM,sigma,alpha):
    K_MN = kernel(X,XM,sigma)
    return np.matmul(K_MN,alpha)

def mse(y1,y2):
    se = 0
    for i,j in zip(y1,y2):
#         se += abs((i-j)/i)
        # se += np.square(i-j)
        se += np.linalg.norm(i-j)**2
    return se/y1.shape[0]


lam = 10**(-4) # variance of data noise
cartpole1 = CartPole()

def gen_train(N,s):
    X = []
    Y = []
    M = 800
    for i in range(N):
        x = random.uniform(-5,5)
        x_dot = random.uniform(-10,10)
        theta = random.uniform(-np.pi,np.pi)
        theta_dot = random.uniform(-15,15)
        act = random.uniform(-20,20)
        Xn = np.array([x,x_dot,theta,theta_dot,act])
        X.append(Xn)
        cartpole1.setState(Xn[:-1])
        cartpole1.performAction(action=Xn[-1])
        Xn_1 = np.array(cartpole1.getState())
        Y.append(Xn_1-Xn[:-1])
    X = np.array(X)
    X_wn = X + np.random.normal(0,s,size=(N,5))
    Y = np.array(Y)
    Y_wn = Y + np.random.normal(0,s,size=(N,4))


    M_ind = random.sample(range(N),M)
    XM = np.array([X[ind] for ind in M_ind])
    XM_wn = np.array([X_wn[ind] for ind in M_ind])
    sigma = [np.std(X[:,i]) for i in range(X.shape[1])]
    K_NM = kernel(X,XM,sigma)
    K_MM = kernel(XM,XM,sigma)
    K_NM_wn = kernel(X_wn,XM_wn,sigma)
    K_MM_wn = kernel(XM_wn,XM_wn,sigma)
    alpha = fit(K_NM,K_MM,lam,Y)
    alpha_wn = fit(K_NM_wn,K_MM_wn,lam,Y_wn)
    Y_predict = predict(X,XM,sigma,alpha)
    Y_predict_wn = predict(X,XM_wn,sigma,alpha_wn)

    return mse(Y_predict,Y), mse(Y_predict_wn,Y)

list_N = []
sigma_list = [0.05, 0.1, 0.2]
m_list = []
m_wnlist = []
N = 1000 

for i in range(4):
    print(N)
    list_N.append(N)
    m = []
    m_wn = []
    for sig in sigma_list:
        a, b = gen_train(N,sig)
        m.append(a)
        m_wn.append(b)
    m_list.append(1.0*sum(m)/len(m))
    m_wnlist.append(m_wn)
    N *= 2
    
m_list = np.array(m_list)
m_wnlist = np.array(m_wnlist)
print(m_list)
print(m_wnlist)

plt.plot(list_N,m_list,label='without noise')
plt.plot(list_N,m_wnlist[:,0],label='observation noise sigma = 0.05')
plt.plot(list_N,m_wnlist[:,1],label='observation noise sigma = 0.1')
plt.plot(list_N,m_wnlist[:,2],label='observation noise sigma = 0.2')
plt.xscale('log')
plt.legend()
plt.show()

