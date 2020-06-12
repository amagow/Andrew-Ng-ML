#Get libraries
import  scipy.optimize as opt
import numpy as np
import pandas
import matplotlib.pyplot as plt

def plotData():
    plt.scatter(X[y==1,0],X[y==1,1], marker='+', c='black',label="Admitted")
    plt.scatter(X[y==0,0],X[y==0,1], marker='.', c='y', label="Not Admitted")
    plt.xlabel("Microchip Test1")
    plt.ylabel("Microchip Test2")
    plt.legend(['y = 1', 'y = 0'])
    
def mapFeature(X1, X2):
    degree = 6
    m = X1.size
    result = np.ones((m,1))
    for i in range(1,degree + 1):
        for j in range(i + 1):
            term1 = X1 ** (i-j)
            term2 = X2 ** (j)
            term  = (term1 * term2).reshape( -1, 1 ) 
            result  = np.append(result, term, 1)
    return result

def sigmoid(z):
    sig = 1./(1. + np.exp(-z)) 
    sig = np.minimum(sig, 0.99999999999999)
    sig = np.maximum(sig, 0.00000000000001)
    return sig
    
def costFunction(theta, X, y):
    y = y.reshape((-1,1))
    theta = theta.reshape((-1,1))
    reg_sum = np.insert(theta[1:],0,0,0)
    cost = (-1./m) * ( y.T@np.log(sigmoid(X@theta)) + (1 - y).T@np.log(1 - sigmoid(X@theta)))
    grad = (1./m)*(X.T@(sigmoid(X@theta) - y))+ (lamda/m)*(reg_sum)
    return np.ndarray.flatten(cost) + (lamda/2*m)*np.sum(np.square(reg_sum)),np.ndarray.flatten(grad)

def gradientFunction(theta, X, y):
    y = y.reshape((-1,1))
    theta = theta.reshape((-1,1))
    reg_sum = np.insert(theta[1:],0,0,0)
    grad = (1./m)*(X.T@(sigmoid(X@theta) - y))+ (lamda/m)*(reg_sum)
    return np.ndarray.flatten(grad)

def plotDecisionBoundary(theta, X, y):
    y = y.reshape((-1,1))
    theta = theta.reshape((-1,1))
    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = mapFeature(np.array(u[i]),np.array(v[j]))@theta
    z = z.T
    plt.contour(u,v,z,[0])
    

#Load data
data = pandas.read_table('ex2data2.txt', sep = ',', names=['Microchip Test1','Microchip Test2','Result'])
X = data.iloc[:,0:2].to_numpy()
y = data.iloc[:,2].to_numpy()


#Plot data
plotData()


##Create 28 features
mappedX = mapFeature(X[:,0], X[:,1])

m, n = mappedX.shape

#Init theta
theta_i = np.zeros(n)

#Set regularisation parameter to 1
lamda = 5.
cost_i, grad_i = costFunction(theta_i, mappedX, y)

options= {'maxiter': 400}

result = opt.minimize(costFunction,
                        theta_i,
                        (mappedX, y),
                        jac=True,
                        method='TNC',
                        options=options)

min_cost, theta = [result.fun, result.x]

plotDecisionBoundary(theta, mappedX, y)