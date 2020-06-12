import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize as opt
from scipy.io import loadmat

def featureScaleAndNormalise(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    
    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm  /= sigma
    return X_norm, mu, sigma

def linearRegCostFunction(X, y, theta, lambda_ = 0.):
    m, n = X.shape
    theta = theta[:,None]
    y = y[:,None]
    h = X@theta

    J = 1 / (2 * m) * np.square(h - y).sum() + lambda_ / (2 * m) * np.square(theta[1:]).sum()
    
    grad = 1 / m * X.T@(h - y)
    grad[1:,:] += (lambda_ / m) * (theta[1:,:]) 
    return J,  grad

def trainLinearReg(X_aug, y, lambda_ = 0., maxiter = 200):

    theta_i = np.zeros(X_aug.shape[1])
    costFunction = lambda t: linearRegCostFunction(X_aug, y, t, lambda_)
    options = {'maxiter': maxiter}
    
    res = opt.minimize(costFunction, theta_i, jac=True, method="TNC" ,options=options)
    return res.x

    
def learningCurve(X_aug, y, Xval, yval, lambda_=0):
    m, n = X.shape
    
    #An array of train and cv costs
    err_train = np.zeros(m)
    err_cv = np.zeros(m)
    
    for i in range(1, m + 1):
        X_samp = X_aug[:i]
        y_samp = y[:i]
        
        #Train Data with regularisation
        theta = trainLinearReg(X_samp, y_samp, lambda_ = lambda_)
        
        #calculate cost wo regularisation
        err_train[i - 1] = linearRegCostFunction(X_samp, y_samp, theta, lambda_ = 0)[0]
        err_cv[i - 1] = linearRegCostFunction(Xval, yval, theta, lambda_ = 0)[0]
    return err_train, err_cv 

def learningCurveAvg(X_aug, y, Xval, yval, lambda_=0):
    m, n = X.shape
    #An array of train and cv costs
    err_train = np.zeros(m)
    err_cv = np.zeros(m)
    
    for i in range(1, m + 1):   
        for j in range(50):
            samps = np.random.choice(m, size = i, replace = True )
            X_samp = X_aug[samps,:]
            y_samp = y[samps]
        
            #Train Data with regularisation
            theta = trainLinearReg(X_samp, y_samp, lambda_ = lambda_)
        
            #calculate cost wo regularisation
            err_train[i - 1] += linearRegCostFunction(X_samp, y_samp, theta, lambda_ = 0)[0]
            err_cv[i - 1] += linearRegCostFunction(Xval, yval, theta, lambda_ = 0)[0]
         
            
    err_train /= 50
    err_cv /= 50
    return err_train, err_cv 
        
def polyFeatures(X, p):
    
    X_poly = np.zeros((X.shape[0], p))
    
    for i in range(p):
        X_poly[:,i] = np.power(X[:,0], i+1)
    
    return X_poly

def plotFit(polyFeatures, min_x, max_x, mu, sigma, theta, p):
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)
    
    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly -= mu
    X_poly /= sigma

    # Add ones
    X_poly = np.concatenate([np.ones((x.shape[0], 1)), X_poly], axis=1)

    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--', lw=2)

def validationCurve(X, y, Xval, yval):
    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
     # You need to return these variables correctly.
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))
    i = 0
    
    for lambda_ in lambda_vec:
        #Train Model with Different values of lambda
        theta = trainLinearReg(X, y, lambda_)
        
        #Calculate the Training Cost and Cross Validation Set Cost for each model
        #No regularisation needed
        error_train[i] = linearRegCostFunction(X, y, theta)[0]
        error_val[i] = linearRegCostFunction(Xval, yval, theta)[0]
        i += 1
        
    return lambda_vec, error_train, error_val

data = loadmat('ex5data1.mat')

X, y = data['X'], data['y'][:, 0]
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]
Xval, yval = data['Xval'], data['yval'][:, 0]

# plt.plot(X, y, 'rx')
# plt.xlabel('Change in water level (x')
# plt.ylabel('Water flowing out of the dam (y)')

theta_i = np.ones((2))
J, grad = linearRegCostFunction(np.insert(X, 0, 1, axis = 1), y, theta_i, 1.)

X_aug = np.hstack(( np.ones((X.shape[0], 1)), X ))
theta_i = np.zeros(X_aug.shape[1])

theta = trainLinearReg(X_aug, y)

plt.plot(X , y, 'rx')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, X_aug@theta, 'b-')

#Learning Curves
Xval_aug = np.hstack(( np.ones((Xval.shape[0], 1)), Xval ))

err_train, err_val = learningCurve(X_aug, y, Xval_aug, yval, lambda_=0)

plt.figure()
plt.plot(np.arange(1, y.size+1), err_train, np.arange(1, y.size+1), err_val, lw=2)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, X.shape[0] + 2, 0, max([np.max(err_train),np.max(err_val)])])

#Polynomial Regression

p = 8

X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureScaleAndNormalise(X_poly)
X_poly = np.insert(X_poly, 0, 1, axis = 1)

X_poly_test = polyFeatures(Xtest, p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.insert(X_poly_test, 0, 1, axis = 1)

X_poly_val = polyFeatures(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.insert(X_poly_val, 0, 1, axis = 1)


for i in [0,1,100]:
    theta_t = trainLinearReg(X_poly, y, lambda_ = i, maxiter = 55) 

    #Plot Training Data and Fit
    plt.figure()
    plt.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')
    plotFit(polyFeatures, np.min(X), np.max(X), mu, sigma, theta_t, p)

    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression Fit (lambda = %f)' % i)
    plt.ylim([-20, 50])


    plt.figure()
    err_train, err_val = learningCurve(X_poly, y, X_poly_val, yval, i)
    plt.plot(np.arange(1, 1+X_poly.shape[0]), err_train, np.arange(1, 1+X_poly.shape[0]), err_val, linewidth=3)

    plt.title('Polynomial Regression Learning Curve (lambda = %f)' % i)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 100])
    plt.legend(['Train', 'Cross Validation'])
    

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

plt.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')

#Lambda of value 3 had the lowest  we use only the validation error to choose best model
best_lambda = np.min(error_val)

#Test model with lowest value of lambda
theta = trainLinearReg(X_poly_test, ytest, best_lambda)
test_error = linearRegCostFunction(X_poly_test, ytest, theta)[0]

#Calculate Learning curve w lambda = 0.01 and taking an avg of 50 time sampling i samples
plt.figure()
err_train, err_val = learningCurveAvg(X_poly, y, X_poly_val, yval, 0.01)
plt.plot(np.arange(1, 1+X_poly.shape[0]), err_train, np.arange(1, 1+X_poly.shape[0]), err_val, linewidth=3)

plt.title('Polynomial Regression Learning Curve (lambda = %f)' % 0.01)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend(['Train', 'Cross Validation'])








