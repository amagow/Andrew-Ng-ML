import numpy as np
from matplotlib import pyplot
# import matplotlib as mpl
from scipy.io import loadmat
# from scipy import  optimize

# ---------------------------------------------------------------------------- #
# Estimating Gaussian - Functions
# ---------------------------------------------------------------------------- #
def scatterPlot(X):
    pyplot.plot(X[:,0], X[:,1], 'bx', ms = 4)
    pyplot.axis([0, 30, 0, 30])
    pyplot.xlabel("Throughput (mb/s)")
    pyplot.ylabel("Latency (ms)")

def estimateGaussian(X):
    m, n = X.shape
    
    mu = 1/m * np.sum(X, axis = 0)
    sigma2 = 1/m * np.sum( np.square(X - mu), axis = 0)
    return mu, sigma2

def multivariateGaussian(X, mu, sigma2):
    k = mu.size
    
    if sigma2.ndim == 1:
        sigma2 = np.diag(sigma2)
    X = X - mu
    p = (2 * np.pi) ** (- k / 2) * np.linalg.det(sigma2) ** (-0.5)\
        * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(sigma2)) * X, axis=1))
    return p

def visualiseFit(X, mu, sigma2):
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    #X1 is plotted on X axis and X2 is plotted on Y axis
    Z = multivariateGaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, sigma2)
    Z = Z.reshape(X1.shape)
    
    pyplot.figure()
    scatterPlot(X)
    
    if np.all(abs(Z) != np.inf):
        #Unsure why these specific values were chosen
        pyplot.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)))
 
def selectThreshold(yval, pval):
    
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    
    for epsilon in np.linspace(1.01*min(pval), max(pval), 1000):
        pvalbinary = pval < epsilon
        tp = np.sum((yval == 1) & (pvalbinary == 1))
        fp = np.sum((yval == 0) & (pvalbinary == 1))
        fn = np.sum((yval == 1) & (pvalbinary == 0))
    
        prec = tp/(tp + fp)
        rec = tp/(tp + fn)
    
        F1 = 2 * prec * rec/ (prec + rec)
        
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #

data = loadmat("ex8data1.mat")
X, Xval, yval = data['X'], data['Xval'], data['yval'][:,0]

scatterPlot(X)

mu , sigma2 = estimateGaussian(X)

p = multivariateGaussian(X, mu, sigma2)

visualiseFit(X,  mu, sigma2)

# ---------------------------------------------------------------------------- #
# CV Set
# ---------------------------------------------------------------------------- #

pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)


print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('Best F1 on Cross Validation Set:  {}'.format(F1))

outliers = p < epsilon

visualiseFit(X,  mu, sigma2)

#Draw Read Circles arounf outliers
pyplot.plot(X[outliers, 0], X[outliers, 1], 'ro', ms=6, mfc='None')


# ---------------------------------------------------------------------------- #
# High Dimensionality Data Set
# ---------------------------------------------------------------------------- #

data = loadmat("ex8data2.mat")
X, Xval, yval = data['X'], data['Xval'], data['yval'][:, 0]

#Apply Same Steps to the Larger DataSet
mu, sigma2 = estimateGaussian(X)

#Training Set
p = multivariateGaussian(X, mu, sigma2)

#  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2)

#  Find the best threshold
epsilon, F1 = selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('Best F1 on Cross Validation Set          : {}'.format(F1))
print('\n# Outliers found: {}'.format(np.sum(p < epsilon)))




















