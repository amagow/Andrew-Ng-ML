# -*- coding: utf-8 -*-
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
from utils import checkNNGradients
import scipy.optimize as optimize
# from matplotlib import gridspec

input_layer_size  = 400;   #20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
  
              # (note that we have mapped "0" to label 10)

def expitGradient(z):
    return expit(z)*(1 - expit(z))

def DisplayData(random_sel, nrows, ncols):
   
    fig = plt.figure(figsize=(6,6))
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(nrows*ncols):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.axis('off')
            plt.imshow(random_sel[i].reshape(20,20).T, cmap = 'Greys', aspect = 'auto')


def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    return np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init

def ForwardPropogation(X, Theta1, Theta2):
    a_1 = np.hstack((np.ones((X.shape[0],1)), X))
    z_2 = a_1@Theta1.T #(5000,25)
    a_2 = expit(z_2) #(5000, 26)
    a_2 = np.insert(a_2, 0, 1, axis = 1)
    
    z_3 = a_2@Theta2.T
    a_3 = expit(z_3)
    # temp = np.zeros_like(a_3)
    # for i in range(a_3.shape[1]):
    #     temp[:,i] = a_3[:, (i - 1) % a_3.shape[1]]
    # a_3 = temp
    return a_1, z_2, a_2, z_3, a_3

def CostFunction(nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, myLambda = 0.):
    m,n = X.shape
    
    encoder = OneHotEncoder(sparse = False)
    y = encoder.fit_transform(y.reshape(-1,1))
    
    #Roll Params
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))
    
    #Get the value of the hypothesis
    h = ForwardPropogation(X, Theta1, Theta2)[4]
    
    J = (-y * np.log(h) - (1 - y) * np.log(1 - h)).sum() / m
    reg_param = (myLambda / (2*m)) * (np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:])))
    return J + reg_param

def BackPropagation(nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, myLambda = 0.):
    
    m = X.shape[0]
    
    #Roll Parameters
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)
    
    #Convert output to a vector for y
    encoder = OneHotEncoder(sparse = False)
    y = encoder.fit_transform(y.reshape(-1,1))
    
    a_1, z_2, a_2, z_3, h = ForwardPropogation(X, Theta1, Theta2)
    
    z_2 = np.insert(z_2, 0, 1, axis = 1)
    
    err_3 = h - y
    err_2 = err_3@Theta2 * expitGradient(z_2)
    
    delta_2 = err_3.T@a_2
    delta_1 = (err_2[:,1:]).T@a_1
    
    D_2, D_1 = delta_2/m , delta_1/m
    
    #Regularized gradients
    D_1[:,1:] += (myLambda / m) * Theta1[:,1:]
    D_2[:,1:] += (myLambda / m) * Theta2[:,1:]
    
    return np.concatenate([D_1.ravel(), D_2.ravel()])
    
    
def nnCostFunction(nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, myLambda = 0.):
    #Implement Backprop Algorithm
    
    m,n = X.shape
    
    cost = CostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, myLambda)
    grad_params = BackPropagation(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, myLambda)
    
    return cost, grad_params


def predict(X, Theta1, Theta2):
    return np.argmax(ForwardPropogation(X, Theta1, Theta2)[4], axis=1)

#1. Load Data
data = loadmat("ex4data1.mat")
X, y = data['X'], data['y']
y = y.reshape(-1)

y[y == 10] = 0

random_sel_indx = np.random.choice(X.shape[0], 100, replace = False)

# Display Data (After lots of meddling got no whitespace bw subplots)
# DisplayData(X[950:1051], 10, 10)

#2. Load Parameters

params = loadmat("ex4weights.mat")

Theta1, Theta2 = params['Theta1'], params['Theta2']

Theta2 = np.roll(Theta2, 1, axis=0)

#Unroll params
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

# Compute Cost (Feed Forward Algo)
J = CostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)
print('Cost at parameters (loaded from ex4weights): {} '.format(J))
print('The cost should be about                   : 0.287629.')

#Compute Cost with Regularisation
J = CostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1.)

print('Cost at parameters (loaded from ex4weights): {}'.format(J))
print('This value should be about                 : 0.383770.')

#Randomly Init Small weight values
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])

#Unregularised checking of gradients
checkNNGradients(nnCostFunction)

#Regularised Gradient Checking
checkNNGradients(nnCostFunction, 3.)

# Also output the costFunction debugging values
debug_J, _  = nnCostFunction(nn_params, input_layer_size,
                          hidden_layer_size, num_labels, X, y, 3)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = {}): {} '.format(3, debug_J))
print('(for lambda = 3, this value should be about 0.576051)')


options = {'maxiter' : 100}

lambda_ = 1.

costFunction = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)  


res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)
nn_params = res.x

optTheta1 = nn_params[:hidden_layer_size*(input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
optTheta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

pred_y = predict(X, optTheta1, optTheta2)
print('Training Set Accuracy: %f' % (np.mean(pred_y == y) * 100))


DisplayData(optTheta1[:, 1:], 5, 5)
















