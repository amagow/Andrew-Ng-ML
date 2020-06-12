#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:17:04 2020

@author: adwithyamagow
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

plt.cla()

def warmUpExercise():
    print(np.identity(5, dtype=int))    
  
def plotData(X, y):
    plt.scatter(X,y, marker="x", c="r")
    plt.xlabel("Population of City in 10,000")
    plt.ylabel("Profit in $10,000")
    axes = plt.gca()
    axes.set_xlim([4,24])
    plt.xticks(np.arange(4, 24+1, 2))
    axes.set_ylim([-5,25])
    plt.show()

def computeCost(X,y,theta):
    m = X.shape[0]
    #n = X.shape[1]
    cost = np.transpose((X@theta - y))@(X@theta - y) / (2 * m)
    return float(cost[0][0])

def gradientDescent(X,y,theta,alpha,iterations):
    m = X.shape[0]
    J_history = np.zeros((iterations, 1))
    for i in range(iterations):
        theta = theta - (alpha/m)* np.transpose(X)@(X@theta - y)
        cost = computeCost(X,y,theta)
        J_history[i][0] = cost
    return theta,  J_history


# -------------------------- ----------------------------------

#warmUpExercise()
#print("Plotting Data....")
df = pd.read_csv('ex1data1.txt', sep=",", names=('Population','Profit'))
X = df.iloc[:,0]
#X.rename("Population")
y = df.iloc[:,1]
y.rename("Profit")
#print(X.size) #Number of training examples
plotData(X,y)

# -------------------------- GRADIENT DESCENT----------------------------------

#Create a dataframe with only 1's for the value of x0
df.insert(0, 'X0', 1)
X = df.iloc[:,0:2]
#Get the two theta parameters
theta = np.zeros((2,1))
iterations = 1500
#Learning Rate
alpha = 0.01
#Had the reshape y from (97,) to (97,1) due to numpy rules
print(computeCost(X.to_numpy(),np.reshape(y.to_numpy(),newshape=(y.size,1)),theta))
#print("".format('%f %f \n', theta[0][0], theta[1][0]))
theta, J = gradientDescent(X.to_numpy(),np.reshape(y.to_numpy(),newshape=(y.size,1)),theta,alpha,iterations)

print("Theta values", theta)
plt.plot(X.iloc[:,1],X@theta)
plt.legend(['Training Data','Linear Regression'])
# ---------------- --------Predict
predict = np.array([1, 1.8])@theta
print("For population = 18,000, we predict a profit of {}".format(10000 * predict[0]))
predict = np.array([1, 2.5])@theta
print("For population = 25,000, we predict a profit of {}".format(10000 * predict[0]))


## ============= Part 4: Visualizing J(theta_0, theta_1) =============
theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1, 4,100)

J_test_plot_vals = np.zeros((theta0_vals.size, theta1_vals.size))

for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        J_test_plot_vals[i][j] = computeCost(X.to_numpy(),y.to_numpy().reshape((y.size, 1)),np.array([[theta0_vals[i]],[theta1_vals[j]]]))

X, Y = np.meshgrid(theta0_vals, theta1_vals)
fig = plt.figure(figsize=plt.figaspect(2.))
ax = fig.add_subplot(2, 1, 1)
ax.grid(True)
ax.contour(X, Y, J_test_plot_vals.T, np.logspace(-2,3,20))
ax.plot(theta[0][0], theta[1][0], c='r', marker="x")
ax = fig.add_subplot(2, 1, 2, projection='3d')
#ax[0] = ax[0].axes(projection='3d')
ax.plot_surface(X, Y, J_test_plot_vals, cmap=cm.coolwarm)