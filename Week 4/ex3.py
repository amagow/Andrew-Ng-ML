from scipy.io import loadmat
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
import scipy.optimize as opt
from scipy.special import expit

def getDatumImage(row, w, h):
    return (row.reshape(w,h)).T
    

def displayData(indices = None):
    """
    Function that picks 100 random rows from X, creates a 20x20 image from each,
    then stitches them together into a 10x10 grid of images, and shows it.
    """
    width, height = 20, 20
    nrows, ncols = 10, 10
    if not indices:
        indices = random.sample(range(X.shape[0]), ncols * nrows)
    picture = np.ones((height*nrows, width*ncols))
    
    irow, icol = 0, 0
    for idx in indices:
        if icol == ncols:
            irow += 1
            icol  = 0
        iimg = getDatumImage(X[idx], width, height)
        picture[irow*height:irow*height+iimg.shape[0],icol*width:icol*width+iimg.shape[1]] = iimg
        icol += 1
    plt.figure(figsize=(6,6))
    img = Image.fromarray( (picture * 255).astype(np.uint8) , mode='L')
    plt.imshow(img,cmap = cm.Greys_r)        

def costFunction(theta, X, y, myLambda = 0.):
    m,n = X.shape
    # y = y.reshape((-1,1))
    # theta = theta.reshape((-1,1))
    # Theta shape: (401, ), X shape: (5000, 401) theta_h shape: (5000, )
    hypothesis = expit(X@theta)
    cost = (-1./m) * ( y.T@np.log(hypothesis) + (1 - y).T@np.log(1 - hypothesis))
    reg_term = (myLambda/2*m) * theta[1:] @ (theta[1:]).T
    return cost + reg_term

def costGradient(theta, X, y, myLambda = 0.):
    m,n = X.shape
    # Theta shape: (401,), X shape: (5000, 401) theta_h shape: (5000,)
    hypothesis = expit(X@theta)
    beta = hypothesis - y
    reg_term = np.insert(theta[1:],0,0,0) * (myLambda/m)
    grad = (1./m)*(X.T@beta) + reg_term
    return grad
   
   
def optimizeTheta(theta, X, y, myLambda=0.):
    y = y.reshape(-1)
    theta = theta.reshape(-1)
  
    result = opt.fmin_cg(costFunction, fprime=costGradient, x0=theta, \
                              args=(X, y, myLambda), maxiter=50, disp=False,\
                              full_output=True)
    return result[0]  

def oneVsAll(X,y, num_labels, myLambda = 0.):
    # Add intercept
    
    m,n = X.shape
    
    theta_i = np.zeros((n))
    all_theta = np.zeros((num_labels, n))
    
    for i in range(num_labels):
        idxClass = i
        if i == 0:
            idxClass = 10
        one_vs_all_Y = (y == idxClass).astype(int)
        all_theta[i,:] = optimizeTheta(theta_i, X, one_vs_all_Y, myLambda)
    return all_theta

def predictOneVsAll(theta, X):
    all_pred = X@theta.T
    return np.argmax(all_pred, axis =1)
    
#Load datafile
data= loadmat('ex3data1.mat')
X, y = data['X'], data['y']
# displayData()

#Vectorize Logistic Regression
myLambda = 0.1;
num_labels = 10

#all_theta = oneVsAll(X, y.reshape(-1,1), num_labels, myLambda)
X = np.hstack(( np.ones(( X.shape[0], 1 )), X ))
all_theta = oneVsAll(X, y, num_labels)

#all_theta_regr = oneVsAll(X, y.reshape(-1,1), num_labels, myLambda, True)


pred_y = predictOneVsAll(all_theta, X)
pred_y[pred_y == 0] = 10
y = y.reshape(-1)


print("Training set accuracy: {}".format(100 * np.mean(y == pred_y)))
