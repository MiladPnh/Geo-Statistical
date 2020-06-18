#Logistic Regression on Synthetic and Real-World Data

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import sklearn.datasets


x = sklearn.datasets.make_classification(n_features=2, n_redundant=0, n_informative=2)
plt.figure()
plt.plot(x[0][np.where(x[1]==0)[0], 0], x[0][np.where(x[1]==0)[0], 1], 'o',label = 'Class 1')
plt.plot(x[0][np.where(x[1]==1)[0], 0], x[0][np.where(x[1]==1)[0], 1], 's', c = 'r',label = 'Class 2')
plt.legend()
plt.show()

w = [np.random.rand()/1000 for j in range(x[0].shape[1])]
eta = 0.01 # Step size of the gradient iteration
print('Please wait until the final result is shown')
for iter in range(1000): # Gradient descent iterations
    dw = np.array([0 for j in range(x[0].shape[1])])
    for j in range(x[0].shape[0]):
        g = np.divide(1,1+np.exp(-np.sum(w*x[0][j])))
        dw = dw + eta*(x[1][j]-g)*x[0][j]
    w = w+dw
    print(w) # The convergence of w can be seen here



# Classify using the result of gradient descent
ygd = [np.sign(sum(np.array(w)*np.array(x[0][j]))) for j in range(x[0].shape[0])]
plt.figure()
plt.suptitle('The result of classification using gradient descent')
plt.plot(x[0][np.where(np.sign(ygd)==-1)[0], 0], x[0][np.where(np.sign(ygd)==-1)[0], 1], 'o', label = 'Class 1')
plt.plot(x[0][np.where(np.sign(ygd)==1)[0], 0], x[0][np.where(np.sign(ygd)==1)[0], 1], 's', c = 'r',label = 'Class 2')
intcpt = np.log(np.divide(np.mean(np.array(x[1])),(1-np.mean(np.array(x[1])))))-np.sum(coef*np.array(x[0]),axis=None)/x[0].shape[0]
ex1 = np.linspace(-2, 3, 3)
ex2 = -(w[0] * ex1 + intcpt) / w[1]
plt.plot(ex1, ex2, color='black', label='decision boundary')
plt.legend()
plt.show()


# Compare the results with the logistic regression found from Python's built-in functions
lrl = LogisticRegression()
x_orig = x[0].copy()
clf = lrl.fit(x_orig, x[1])
yhatl = lrl.predict(x_orig)
plt.figure()
plt.suptitle('The result of classification using Pyhton\'s built-in functions')
plt.plot(x_orig[np.where(yhatl==0)[0], 0], x_orig[np.where(yhatl==0)[0], 1], 'o', label = 'Class 1')
plt.plot(x_orig[np.where(yhatl==1)[0], 0], x_orig[np.where(yhatl==1)[0], 1], 's', c = 'r',label = 'Class 2')
from sklearn.multiclass import OneVsRestClassifier
classifier = OneVsRestClassifier(LogisticRegression(penalty='l1')).fit(x_orig, x[1])
coef = classifier.coef_
intercept = classifier.intercept_
ex1 = np.linspace(-2, 3, 3)
ex2 = -(coef[:, 0] * ex1 + intercept[:, 0]) / coef[:,1]
ex1 = np.linspace(-2, 3, 3)
ex2 = -(coef[:, 0] * ex1 + intercept[:, 0]) / coef[:,1]
plt.plot(ex1, ex2, color='black', label='decision boundary')
plt.legend()
plt.show()

print([coef,w]) # Comparing the coefficients of the LR using gradient descent and built-in Python functions

# Comparing the resulting coefficients of each method
print(''.join(['Python built-in: ',''.join(np.array_str(coef[0]))]))
print(''.join(['Gradient Descent: ',np.array_str(w)]))

#################################################################
#Python built-in: [-0.38124485  2.34903593]
#Gradient Descent: [-0.57499293  2.6584022 ]

