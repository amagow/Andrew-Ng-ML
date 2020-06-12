#10 is actually mapped to 0

from scipy.io import loadmat
import numpy as np
from scipy.special import expit


def predict(Theta1, Theta2, X):
    a_2 = expit(X@Theta1.T)
    a_2 = np.hstack(( np.ones(( a_2.shape[0], 1 )), a_2 ))
    
    a_3 = expit(a_2@Theta2.T)
    output = a_3
    
    #Theta1 and Theta2 are one basded indexing
    return np.argmax(output, axis = 1) + 1


input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

#Load Data
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y']

#Select random int for display
sel = X[np.random.randint(0, X.shape[0], 100), :]

weights = loadmat("ex3weights.mat")
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

X = np.hstack(( np.ones(( X.shape[0], 1 )), X ))
pred_y = predict(Theta1, Theta2, X)

y = y.reshape(-1)

print("Training Set Accuracy: {}".format(100 * np.mean(pred_y == y)))
