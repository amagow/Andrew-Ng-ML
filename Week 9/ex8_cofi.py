import numpy as np
from matplotlib import pyplot
from scipy.io import loadmat
from scipy import optimize


# ---------------------------------------------------------------------------- #
# Check Cost Function
# ---------------------------------------------------------------------------- #
def computeNumericalGradient(J, theta, e=1e-4):
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*e)
    return numgrad

def checkCostFunction(cofiCostFunc, lambda_=0.):
    # Create small problem
    X_t = np.random.rand(4, 3)
    theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, theta_t.T)
    Y[np.random.rand(*Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # Run Gradient Checking
    X = np.random.randn(*X_t.shape)
    theta = np.random.randn(*theta_t.shape)
    num_movies, num_users = Y.shape
    num_features = theta_t.shape[1]

    params = np.concatenate([X.ravel(), theta.ravel()])
    numgrad = computeNumericalGradient(
        lambda x: cofiCostFunc(x, Y, R, num_users, num_movies, num_features, lambda_), params)

    cost, grad = cofiCostFunc(params, Y, R, num_users,num_movies, num_features, lambda_)

    print(np.stack([numgrad, grad], axis=1))
    print('\nThe above two columns you get should be very similar.'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)')

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your cost function implementation is correct, then '
          'the relative difference will be small (less than 1e-9).')
    print('\nRelative Difference: %g' % diff)
# ---------------------------------------------------------------------------- #
# Collaborative Filtering Algorithm
# ---------------------------------------------------------------------------- #

def cofiCostFunc(params, Y, R, num_users, num_movies,
                      num_features, lambda_=0.0):
    
    #Unroll Parameters
    X = params[:num_features * num_movies].reshape(num_movies, num_features)
    theta = params[num_features * num_movies:].reshape(num_users, num_features)
    
    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)
    
    J = 0.5 * np.sum(np.square((X@theta.T - Y)*R)) \
            + lambda_/2 *(np.sum(np.square(theta)) + np.sum(np.square(X)))
            
    X_grad = (((X@theta.T - Y)*R)@theta) + lambda_ * X
    
    theta_grad = (((X@theta.T - Y)*R).T@X) + lambda_ * theta
    return J, np.concatenate([X_grad.ravel(), theta_grad.ravel()])

def normalizeRatings(Y, R):
    Y_mean = np.mean(Y, axis=1)
    Y_norm = Y - Y_mean[:,None]
    return Y_norm, Y_mean
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #

data = loadmat("ex8_movies.mat")
Y, R = data['Y'], data['R']

# From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): {:.1f} / 5'.format(np.mean(Y[0, R[0, :] == 1])))

# We can "visualize" the ratings matrix by plotting it with imshow
pyplot.figure(figsize=(8, 8))
pyplot.imshow(Y)
pyplot.ylabel('Movies')
pyplot.xlabel('Users')
pyplot.grid(False)

#  Load pre-trained weights (X, theta, num_users, num_movies, num_features)
data = loadmat("ex8_movieParams.mat")
X, theta, num_users, num_movies, num_features = data['X'],\
        data['Theta'], data['num_users'], data['num_movies'], data['num_features']

#  Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3

X = X[:num_movies, :num_features]
theta = theta[:num_users, :num_features]
Y = Y[:num_movies, 0:num_users]
R = R[:num_movies, 0:num_users]

#  Evaluate cost function
J, _ = cofiCostFunc(np.concatenate([X.ravel(), theta.ravel()]),
                    Y, R, num_users, num_movies, num_features)
           
print('Cost at loaded parameters:  {:.2f} \n(this value should be about 22.22)'.format(J))

checkCostFunction(cofiCostFunc)

J, _ = cofiCostFunc(np.concatenate([X.ravel(), theta.ravel()]),
                    Y, R, num_users, num_movies, num_features, 1.5)
           
print('Cost at loaded parameters (lambda = 1.5): {:.2f}'.format(J))
print('              (this value should be about 31.34)')

checkCostFunction(cofiCostFunc, 1.5)

# ---------------------------------------------------------------------------- #
# Learn Movie Reccomendations
# ---------------------------------------------------------------------------- #

with open("movie_ids.txt",  encoding='ISO-8859-1') as fid:
    movies = fid.readlines()

movieList = []
for movie in movies:
    movieList.append(''.join(movie.split()[1:]).strip())

n_m = len(movieList)

#  Initialize my ratings
my_ratings = np.zeros(n_m)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
# Note that the index here is ID-1, since we start index from 0.
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[1] = 3.5
my_ratings[10] = 4.5
my_ratings[28] = 3.5
my_ratings[49] = 4
my_ratings[63] = 4.5
my_ratings[68] = 4
my_ratings[70] = 4
my_ratings[71] = 3.5
my_ratings[81] = 4.5
my_ratings[82] = 3.5
my_ratings[93] = 4
my_ratings[94] = 4
my_ratings[95] = 4
my_ratings[98] = 4
my_ratings[126] = 4.5
my_ratings[143] = 4.5
my_ratings[173] = 4
my_ratings[180] = 3
my_ratings[184] = 3.5
my_ratings[186] = 4
my_ratings[203] = 4
my_ratings[248] = 3
my_ratings[251] = 3.5
my_ratings[256] = 4.5
my_ratings[260] = 4
my_ratings[312] = 4.5
my_ratings[406] = 3.5
my_ratings[598] = 4
my_ratings[767] = 3.5
my_ratings[819] = 4.5



print('\nNew user ratings:')
print('-----------------')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d stars: %s' % (my_ratings[i], movieList[i]))


#  Now, train the collaborative filtering model on a movie rating 
#  dataset of 1682 movies and 943 users

#  Load data
data = loadmat("ex8_movies.mat")
Y, R = data['Y'], data['R']

#Add my ratings
Y = np.concatenate([Y, my_ratings[:,None]], axis = 1)
R = np.concatenate([R, my_ratings[:,None] > 0], axis = 1)

#  Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
num_movies, num_users = Y.shape
num_features = 10

# Set Initial Parameters (theta, X)
X = np.random.randn(num_movies, num_features)
theta = np.random.randn(num_users, num_features)

init_params = np.concatenate([X.ravel(), theta.ravel()])

# Set options for scipy.optimize.minimize
options = {'maxiter': 100}

# Set Regularization
lambda_ = 10
res = optimize.minimize(lambda x: cofiCostFunc(x, Ynorm, R, num_users,
                                               num_movies, num_features, lambda_),
                        init_params,
                        method='TNC',
                        jac=True,
                        options=options)
return_param = res.x

# Unfold the returned theta back into U and W
X = return_param[:num_movies*num_features].reshape(num_movies, num_features)
theta = return_param[num_movies*num_features:].reshape(num_users, num_features)

print('\nRecommender system learning completed.')

p = X@theta.T
my_predictions = p[:, -1] + Ymean

ix = np.argsort(my_predictions)[::-1]

print('Top recommendations for you:')
print('----------------------------')
for i in range(10):
    j = ix[i]
    print('Predicting rating %.1f for movie %s' % (my_predictions[j], movieList[j]))

print('\nOriginal ratings provided:')
print('--------------------------')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s' % (my_ratings[i], movieList[i]))