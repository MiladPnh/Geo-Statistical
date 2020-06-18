#Naive Bayes Spam Filter

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


#We first separate the data and the classes

data = np.genfromtxt('spambase_train.csv', delimiter=',')
x = data[:, 0:-1]
y = data[:, -1]

#We then divide the data into the training chunk and the test chunk. We establish the naive Bayes classifier
#with the training chunk, and then try to classify the test chunk. The result of classification of test chunk is
#then compared to the original classes corresponding to the test chunk, which gives us the classification error.
#By permutating the training chunk (and test chunk) throughout the whole dataset, we can calculate five classification error values.
#These values ar then averaged, resulting in the 5-fold cross-validation error.

error = 0
gnb = GaussianNB()
k = 1
clf = svm.SVC(kernel='linear', C=1)
for k in range(5):
    x_testfold = x[k * 734:(k + 1) * 734, :]  # Testing Chunk
    y_testfold = y[k * 734:(k + 1) * 734]  # Testing Chunk Classes
    x_trainfold = x[[num for num in range(len(x)) if num not in range(k * 734, (k + 1) * 734)], :]  # Training Chunk
    y_trainfold = y[[num for num in range(len(x)) if num not in range(k * 734, (k + 1) * 734)]]  # Training Chunk Classes
    gnb.fit(x_trainfold, y_trainfold)
    yhat = gnb.predict(x_testfold)
    error = error + gnb.score(x_testfold, y_testfold)  # np.sum(1.*(yhat != y_testfold))/len(yhat)

error = error/5 # 5-fold Cross-Validation Error
print("Method 1: 5-fold cross-validation score is", 1-error)


# Alternate solution with shuffling data at each iteration

cv1 = ShuffleSplit(n_splits=5, test_size=0.2)
scores = cross_val_score(gnb,x,y,cv = cv1)
print("Method 2: 5-fold cross-validation score is", 1-scores.mean())

# Note: To avoid shuffling the data, the variable \"cv1\" can be changed to simply a constant number, (in this case cv1 = 5 for 5-fold cross-validation)



