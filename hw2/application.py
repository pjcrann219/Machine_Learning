import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.array(y).T
X = np.array(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################

def train_and_test(Xtrain, Ytrain, Xtest, Ytest, alpha, n_epoch):
    w = train(Xtrain, Ytrain, alpha, n_epoch)
    yhat = compute_yhat(Xtest, w)
    L = compute_L(yhat,Ytest)
    return L

n_epochs = np.array([10, 100, 1000, 10000, 100000])
for n_epoch in n_epochs:
    L = train_and_test(Xtrain, Ytrain, Xtest, Ytest, 0.01, n_epoch)
    print(f"n_epoch: {n_epoch}\tL: {L}")

alphas = np.array([.1, .01, .001, .0001])
for alpha in alphas:
    L = train_and_test(Xtrain, Ytrain, Xtest, Ytest, alpha, 100000)
    print(f"alpha: {alpha}\tL: {L}")

#########################################

