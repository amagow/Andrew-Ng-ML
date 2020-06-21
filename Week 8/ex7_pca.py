import numpy as np
# import re
from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
# import matplotlib as mpl
# from IPython.display import HTML, display, clear_output
# from scipy import optimize
from scipy.io import loadmat
from sklearn import preprocessing

# ---------------------------------------------------------------------------- #
# PCA Algorithm
# ---------------------------------------------------------------------------- #

def pca(X):
    m,n = X.shape
    
    sigma = 1/m * (X.T@X)
    U, S, V = np.linalg.svd(sigma)
    
    return U, S

def projectData(X, U, K):
    return X@U[:,:K]

def recoverData(Z, U, K):
    return Z@U[:,:K].T

# ---------------------------------------------------------------------------- #
# 
# ---------------------------------------------------------------------------- #

data = loadmat("ex7data1.mat")
X = data['X']

#  Visualize the example dataset
pyplot.plot(X[:, 0], X[:, 1], 'wo', ms=5, mec='b', mew=1)
pyplot.axis([0.5, 6.5, 2, 8])
pyplot.gca().set_aspect('equal')
pyplot.grid(False)

X_scaled = preprocessing.scale(X)
mu = X.mean(axis = 0)

#Run PCA
U, S = pca(X_scaled)

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
for i in range(2):
    pyplot.arrow(mu[0], mu[1], 1.5 * S[i]*U[0, i], 1.5 * S[i]*U[1, i],
             head_width=0.25, head_length=0.2, fc='k', ec='k', lw=2, zorder=1000)

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_scaled, U, K)
X_rec  = recoverData(Z, U, K)

#  Plot the normalized dataset (returned preprocessing.scale)

pyplot.figure(figsize=(5,5))
pyplot.plot(X_scaled[:, 0], X_scaled[:, 1], 'wo', ms=5, mec='b', mew=1)
pyplot.gca().set_aspect('equal')
pyplot.grid(False)
pyplot.axis([-3, 2.75, -3, 2.75])

#Plot the Recovered Data Set after PCA
pyplot.plot(X_rec[:, 0], X_rec[:, 1], 'wo', ms=4, mec='r', mew=1.5)

# Draw lines connecting the projected points to the original points
for i in range(X.shape[0]):
    pyplot.plot([X_scaled[i,0],X_rec[i, 0]], [X_scaled[i, 1], X_rec[i, 1]], '--k', lw = 1)
    
    
# ---------------------------------------------------------------------------- #
# PCA on faces
# ---------------------------------------------------------------------------- #   

def displayData(X, example_width=None, figsize=(10, 10)):
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')
    
    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = int(n / example_width)
    
    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    
    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()
    
    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_height, example_width, order='F'), cmap='gray')
        ax.axis('off')
    
data = loadmat("ex7faces.mat")
X = data['X']

displayData(X[:100, :], figsize=(8, 8))  
    
X_scaled = preprocessing.scale(X)

U, S = pca(X_scaled)
    
#  Visualize the top 36 eigenvectors found
displayData(U[:, :100].T, figsize=(8, 8))    

K = 350
Z = projectData(X_scaled, U, K)    

#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed
X_rec  = recoverData(Z, U, K)    
    
# Display normalized data
displayData(X_scaled[:100, :], figsize=(6, 6))
pyplot.gcf().suptitle('Original faces')

# Display reconstructed data from only k eigenfaces
displayData(X_rec[:100, :], figsize=(6, 6))
pyplot.gcf().suptitle('Recovered faces')    
    