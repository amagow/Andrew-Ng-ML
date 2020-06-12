#Get libraries
import  scipy.optimize as opt
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.special import expit


def plotData():
    plt.scatter(X[y==1,0],X[y==1,1], marker='+', c='black',label="Admitted")
    plt.scatter(X[y==0,0],X[y==0,1], marker='.', c='y', label="Not Admitted")
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend(['Admitted', 'Not Admitted'])

def sigmoid(z):
    sig = 1./(1. + np.exp(-z)) 
    sig = np.minimum(sig, 0.9999999)
    sig = np.maximum(sig, 0.0000001)
    return sig

def costFunction(theta, X, y):
    cost = (-1./m) * ( y[:,None].T@np.log(sigmoid(X@theta[:,None])) + (1 - y[:,None]).T@np.log(1 - sigmoid(X@theta[:,None])))
    return np.ndarray.flatten(cost)

def costFunctionDer(theta, X, y):
    grad = (1./m)*(X.T@(sigmoid(X@theta[:,None]) - y[:,None]))
    return np.ndarray.flatten(grad)

def predict(theta, X):
    return (X@theta >= 0).astype('int')

#Load Data
data = pandas.read_table('ex2data1.txt', sep=',', names=['Exam 1 Score','Exam 2 Score','Admittance'] )
X = data.iloc[:,0:2].to_numpy()
y = data.iloc[:,2].to_numpy()

#Plot Data
plotData()

#Compute Cost and Gradient

#Add intercept
X = np.insert(X, 0, 1, 1)

m, n = X.shape

#Initialise thetas
theta_i = np.zeros((n))


cost = costFunction(theta_i, X, y)
grad = costFunctionDer(theta_i, X, y)

#Optimise Algorithm for theta
result = opt.fmin_bfgs(costFunction, theta_i, costFunctionDer, args=(X,y), full_output = True, maxiter=400, retall=True)

theta, cost_min = result[0:2]

#Make a decision Boundary vv smart, take the min and max range of x
# and then calculate the value of score 2 from it as theta[0] + x1theta[1] + x2theta[2] = 0
boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
plt.legend()

#Predict
p = predict(theta, X)
print("Train Accuracy {}".format(100*np.mean(p==y)))
