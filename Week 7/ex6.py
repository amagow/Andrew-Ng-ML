import numpy as np
# import re
from matplotlib import pyplot as plt
# from scipy import optimize
from scipy.io import loadmat
from sklearn import svm
import math

def scatterPlot(X, y):
    #Plot data
    plt.figure()
    true_idx = (y[:] == 1)
    plt.scatter(X[true_idx,0], X[true_idx,1], marker = "x", color="royalblue", linewidth = 4)
    plt.scatter(X[~true_idx,0], X[~true_idx,1], marker = "o", color="orange")

def makeMeshgrid(x, y, h = 0.2):
    x_min, x_max = x.min() - 0.01, x.max() + 0.01
    y_min, y_max = y.min() - 0.01, y.max() + 0.01
    return np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

def plotContours(xx, yy,model, alpha):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha= alpha)

def SVCDecisionBoundary(X,y, model,xlim = (0,0), ylim = (0,0), h = 0.2, alpha=0.8):
    #Create a new plot
    plt.figure()
    title = ('Decision Boundary of Linear SVC ')
    
    #Make a meshgrid for the plot
    
        
    
    #Plot a contour plot
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    if xlim == (0,0) and ylim == (0,0):
        xx, yy = makeMeshgrid(X[:,0],X[:,1], h)
        plotContours(xx,yy, model, alpha)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
    else:
        xx, yy = makeMeshgrid(np.array(xlim),np.array(ylim), h)
        plotContours(xx,yy, model, alpha)
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])
    plt.title(title)
    
def gaussianKernel(x1, x2, sigma):
    return math.exp(-np.sum((x1-x2)**2)/(2 * (sigma ** 2)))

def dataset3Params(X, y, Xval, yval):
    global pred_error
    values = np.linspace(1, 20, 100)
    pred_error = np.zeros((len(values), len(values)))
    
    i, j = 0, 0
    for C in values:
        j = 0
        for sigma in values:
            model = svm.SVC(C = C, gamma = sigma ).fit(X,y)
            pred = model.predict(Xval )
            pred_error[i][j] = np.mean(pred != yval)
            j +=1
        i += 1
    return values[np.argmin(np.min(pred_error, axis=0))], values[np.argmin(np.min(pred_error, axis=1))]

#Load Data
data = loadmat("ex6data1.mat")
X = data['X']
y = data['y'].reshape(-1)

scatterPlot(X, y)

#Try Linear Kernel SVM

linModelSVM = svm.SVC(C = 1.0, kernel="linear").fit(X,y)
linModelSVM2 = svm.SVC(C = 100.0, kernel="linear").fit(X,y)

SVCDecisionBoundary(X, y, linModelSVM, h = 0.01)
SVCDecisionBoundary(X, y, linModelSVM2, h = 0.01)

#Test Gaussian Kernel
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

sim = gaussianKernel(x1, x2, sigma)
print(sim) #Kernel is correct

data2 = loadmat("ex6data2.mat")

X2 = data2['X']
y2 = data2['y'].reshape(-1)

#Train Gaussian Model SVM
rbfModelSVM = svm.SVC(C = 1., kernel="rbf", gamma = 100 ).fit(X2,y2)
scatterPlot(X2, y2)
SVCDecisionBoundary(X2, y2, rbfModelSVM, h = 0.005)

#Load Dataset 3
data3 = loadmat("ex6data3.mat")
X, y, Xval, yval = data3['X'], data3['y'][:, 0], data3['Xval'], data3['yval'][:, 0]

scatterPlot(X, y)
C, sigma = dataset3Params(X, y, Xval, yval)

model = svm.SVC(C = C, gamma = sigma ).fit(X,y)
SVCDecisionBoundary(X, y, model, h = 0.001, alpha=0.6)


