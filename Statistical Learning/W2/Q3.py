#Density Estimation in Practice

#Note : In this excercise I decied to model the posterior probability using KNN method.
#  I set k = 5, and used 0.1 of the whole data set as training data. The rest is used for test data.

def gen_cb(N, a, alpha):
    """
    N: number of points on the checkerboard
    a: width of the checker board (0<a<1)
    alpha: rotation of the checkerboard in radians
    """
    d = np.random.rand(N, 2).T
    d_transformed = np.array([d[0]*np.cos(alpha)-d[1]*np.sin(alpha),
                              d[0]*np.sin(alpha)+d[1]*np.cos(alpha)]).T
    s = np.ceil(d_transformed[:,0]/a)+np.floor(d_transformed[:,1]/a)
    lab = 2 - (s%2)
    data = d.T
    return data, lab

# X, y = gen_cb(500, .5, 0)
X, y = gen_cb(5000, .25, 3.14159/4)
plt.figure()
plt.suptitle('Complete Data')
plt.plot(X[np.where(y==1)[0], 0], X[np.where(y==1)[0], 1], 'o',label = 'Class 1')
plt.plot(X[np.where(y==2)[0], 0], X[np.where(y==2)[0], 1], 's', c = 'r',label = 'Class 2')
plt.legend()
plt.show()

# Split the data to training and test data
np.random.seed(0)
indices = np.random.permutation(len(X))
X_train = X[indices[-len(X)/10:]]
y_train = y[indices[-len(X)/10:]]
X_test  = X[indices[1:1-len(X)/10]]
y_test  = y[indices[1:1-len(X)/10]]
# Use nearest neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30,
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
yhat = knn.predict(X_test)
plt.figure()
plt.suptitle('Training Data')
plt.plot(X_train[np.where(y_train==1)[0], 0], X_train[np.where(y_train==1)[0], 1], 'o', label = 'Class 1')
plt.plot(X_train[np.where(y_train==2)[0], 0], X_train[np.where(y_train==2)[0], 1], 's', c = 'r',label = 'Class 2')
plt.legend()
plt.show()
plt.figure()
plt.suptitle('Test Data')
plt.plot(X_test[np.where(y_test==1)[0], 0], X_test[np.where(y_test==1)[0], 1], 'o', label = 'Class 1')
plt.plot(X_test[np.where(y_test==2)[0], 0], X_test[np.where(y_test==2)[0], 1], 's', c = 'r',label = 'Class 2')
plt.legend()
plt.show()
plt.figure()
plt.suptitle('Result of Classification')
plt.plot(X_test[np.where(yhat==1)[0], 0], X_test[np.where(yhat==1)[0], 1], 'o', label = 'Class 1')
plt.plot(X_test[np.where(yhat==2)[0], 0], X_test[np.where(yhat==2)[0], 1], 's', c = 'r',label = 'Class 2')
plt.legend()
plt.show()



