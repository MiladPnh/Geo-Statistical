from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import sys
import numpy as np
import scipy as sp
import matplotlib
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn
from sklearn.preprocessing import scale, MinMaxScaler


df = pd.read_csv('11.csv').replace(' ', '')
df = df.apply(pd.to_numeric)

X = df.drop('Class', axis=1)
X = X.drop('Area Growth Rate', axis=1)
y = df['Area Growth Rate']
y = np.asarray(y)
y = (y-min(y))/max(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)


sklearn_pca = PCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_train)
gmm = GaussianMixture(n_components=5, covariance_type='full').fit(Y_sklearn)
prediction_gmm = gmm.predict(Y_sklearn)
probs = gmm.predict_proba(Y_sklearn)


#Ploting

centers = np.zeros((5,2))
for i in range(5):
    density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(Y_sklearn)
    centers[i, :] = Y_sklearn[np.argmax(density)]

plt.figure(figsize = (10,10))
plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=prediction_gmm ,s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);
plt.show()



