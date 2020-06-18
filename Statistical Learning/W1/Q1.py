#HalfMoon Data Generator and Linear Classifier

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import sklearn.datasets
from sklearn.multiclass import OneVsRestClassifier


#In the below figure, a halfmoon dataset with 1000 points is generated, which are corrupted by a white Gaussian noise whose standard deviation is 0.3
nsamples = 1000
x = sklearn.datasets.make_moons(nsamples,shuffle = True,noise = 0.3)
plt.figure()
plt.plot(x[0][np.where(x[1]==0)[0], 0], x[0][np.where(x[1]==0)[0], 1], 'o',label = 'Class 1')
plt.plot(x[0][np.where(x[1]==1)[0], 0], x[0][np.where(x[1]==1)[0], 1], 's', c = 'r',label = 'Class 2')
plt.legend()
plt.show()

#Build linear classifier

lrl = LogisticRegression()
x_orig = x[0].copy()
clf = lrl.fit(x_orig, x[1])
yhatl = lrl.predict(x_orig)
plt.figure()
plt.plot(x_orig[np.where(yhatl==0)[0], 0], x_orig[np.where(yhatl==0)[0], 1], 'o', label = 'Class 1')
plt.plot(x_orig[np.where(yhatl==1)[0], 0], x_orig[np.where(yhatl==1)[0], 1], 's', c = 'r',label = 'Class 2')
classifier = OneVsRestClassifier(LogisticRegression(penalty='l1')).fit(x_orig, x[1])
coef = classifier.coef_
intercept = classifier.intercept_
ex1 = np.linspace(-2, 3, 3)
ex2 = -(coef[:, 0] * ex1 + intercept[:, 0]) / coef[:,1]
plt.plot(ex1, ex2, color='black', label='decision boundary')
plt.legend()
plt.show()

#It can be seen that using a linear classifier while the dimensionality of feature vector is intact does not classify the data as expected.
#Hence, we used a feature transformation that raises the dimensionality of feature vector from 2 to 55. This is done due to Cover's theorem
#which suggests the possible existence of a linear classifier when the feature vector is transformed to have more dimensions.


poly = PolynomialFeatures(degree=9)
x_new = poly.fit_transform(x_orig)

#Print the size of the original feature vector
print(x_orig.shape)
#Print the size of the transofrmed feature vector
print(x_new.shape)

#We now establish a linear classifier (i.e., logistic regression) that classifies the transofrmed feature vector.
lr = LogisticRegression()
lr.fit(x_new, x[1])
yhat = lr.predict(x_new)
plt.figure()
plt.plot(x_orig[np.where(yhat==0)[0], 0], x_orig[np.where(yhat==0)[0], 1], 'o',label = 'Class 1')
plt.plot(x_orig[np.where(yhat==1)[0], 0], x_orig[np.where(yhat==1)[0], 1], 's', c = 'r', label = 'Class 2')
plt.legend()
plt.show()

#It can be seen that although in previous figure the linear classifier was not performing well, when the feature vector has more dimensions,
#even a simple classifier such as logisitic regression can classify the data noticeabley better than the previous attempt.
#However, the dimensionality of data is increased, which may become expensive, as the the computational cost increases.

