#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:46:18 2020

@author: adwithyamagow
"""
#================ Part 1: Feature Normalization ================

import numpy as np
import matplotlib.pyplot as plt

def plotConvergence(J):
    plt.figure()
    plt.plot(range(len(J)), J)
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")

def featureNormalize(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X = (X - mu.T)/sigma.T
    return X, mu, sigma

def computeCost(X,y,theta):
    cost = np.transpose((X@theta - y))@(X@theta - y) / (2 * m)
    return float(cost[0][0])

def gradientDescentMulti(X, y, theta, alpha, iterations):
    J_history = np.zeros((iterations, 1))
    for i in range(iterations):
        theta = theta - (alpha/m)* np.transpose(X)@(X@theta - y)
        cost = computeCost(X,y,theta)
        J_history[i][0] = cost
    return theta,  J_history

data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, [0,1]]
y = data[:, 2]
m = y.size

#We do not use feature scaling on the intercet term
X, mu, sigma = featureNormalize(X)
X = np.insert(X,0, 1,axis = 1)

# ================ Part 2: Gradient Descent ================

alpha = 0.3
iterations = 1500

theta = np.zeros((3, 1))

theta, J_history = gradientDescentMulti(X, y.reshape((y.size, 1)), theta, alpha, iterations)
plotConvergence(J_history)

print(theta)
ytest = np.array([1650.,3.])
ytestscaled = [(ytest[x] - mu[x])/sigma[x] for x in range(len(ytest))]
ytestscaled.insert(0,1)
print(float(theta.T@ytestscaled))
#  ================ Part 3: Normal Equations ================

X2 = data[:, [0,1]]
X2 = np.insert(X2,0,1,axis=1)

theta_normal = (np.linalg.inv((X2.T@X2))@X2.T)@y
print(float(theta_normal.T@[1, 1650., 3]))

