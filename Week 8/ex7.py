import numpy as np
# import re
from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
# from IPython.display import HTML, display, clear_output
# from scipy import optimize
from scipy.io import loadmat
try:
    pyplot.rcParams["animation.html"] = "jshtml"
except ValueError:
    pyplot.rcParams["animation.html"] = "html5"

# ---------------------------------------------------------------------------- #
#Implementing K Means
# ---------------------------------------------------------------------------- #
    
def findClosestCentroids(X, centroids):
    K = centroids.shape[0]
    m,n = X.shape
    idx = np.zeros(m, dtype=int)
    
    for i in range(m):
        idx[i] = np.argmin([np.sum(np.square(X[i] - centroids[j])) for j in range(K)], axis = 0) 
    return idx

def computeCentroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for i in range(K):
        centroids[i] = np.mean(X[idx==i], axis = 0)
    return centroids

def kMeansInitCentroids(X, K):
    m,n = X.shape
    centroids = np.zeros((K,n))
    centroids = X[np.random.choice(m,size = K, replace = False), :]
    return centroids

def plotProgresskMeans(i, X, centroid_history, idx_history):
     K = centroid_history[0].shape[0]
     pyplot.gcf().clf()
     cmap = pyplot.cm.rainbow
     norm = mpl.colors.Normalize(vmin = 0, vmax = K - 1)
     
     for j in range(K):
         current = np.stack([c[j, :] for c in centroid_history[:i+1]], axis=0)
         
         pyplot.plot(current[:, 0], current[:, 1],
                    '-Xk',
                    mec='k',
                    lw=2, #Line Width
                    ms=10, #Marker Size
                    mfc=cmap(norm(j)), #Marker Face Colour
                    mew=2) #Marker Edge Width

         pyplot.scatter(X[:, 0], X[:, 1],
                       c=idx_history[i],
                       cmap=cmap,
                       marker='o',
                       s=8**2, #Marker Size in points^2
                       linewidths=1,)
     pyplot.grid(False)
     pyplot.title('Iteration number %d' % (i+1))
     
def runkMeans(X, centroids, findClosestCentroids, computeCentroids,
              max_iters=10, plot_progress=False):
    global centroid_history
    K = centroids.shape[0]
    idx = None
    idx_history = []
    centroid_history = []
    
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        
        if plot_progress:
            idx_history.append(idx)
            centroid_history.append(centroids)
        
        centroids = computeCentroids(X, idx, K)
        
    if plot_progress:
        fig = pyplot.figure()
        anim = FuncAnimation(fig, plotProgresskMeans,
                             frames=max_iters,
                             interval=500,
                             repeat_delay=2,
                             fargs=(X, centroid_history, idx_history))
        return centroids, idx, anim
        
    return centroids, idx

    

# ---------------------------------------------------------------------------- #
#Tests for The K-Means Functions
# ---------------------------------------------------------------------------- #

# data = loadmat("ex7data2.mat")
# X = data['X']

# Select an initial set of centroids
# K = 3   # 3 Centroids
# initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# idx = findClosestCentroids(X, initial_centroids)

# print('Closest centroids for the first 3 examples:')
# print(idx[:3])
# print('(the closest centroids should be 0, 2, 1 respectively)')

# centroids = computeCentroids(X, idx, K)
# print('Centroids computed after initial finding of closest centroids:')
# print(centroids)
# print('\nThe centroids should be')
# print('   [ 2.428301 3.157924 ]')
# print('   [ 5.813503 2.633656 ]')
# print('   [ 7.119387 3.616684 ]')

# ---------------------------------------------------------------------------- #
#Run K-Means On data Set
# ---------------------------------------------------------------------------- #
centroid_history = None
# Load an example dataset
data = loadmat("ex7data2.mat")
X = data['X']

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples.
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means

centroids, idx, anim = runkMeans(X, initial_centroids,
                                       findClosestCentroids, computeCentroids, max_iters, True)

# ---------------------------------------------------------------------------- #
#Run K-Means On Picture to use K-Means on
# ---------------------------------------------------------------------------- #

K = 10
max_iters = 10

# Load an image of a bird
A = mpl.image.imread("test_colorful_image.jpg")

# #Normalise Data
A_ = A/255

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = A_.reshape(-1, 3)

#Randomply Initialise Centroids
initial_centroids = kMeansInitCentroids(X, K)


centroids, idx = runkMeans(X, initial_centroids,
                                       findClosestCentroids, computeCentroids, max_iters, False) 


X_recovered = centroids[idx,:].reshape(A.shape)

fig, ax = pyplot.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(A)
ax[0].set_title('Original')
ax[0].grid(False)

# Display compressed image, rescale back by 255
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed, with %d colors' % K)
ax[1].grid(False)




























