from __future__ import print_function
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
plotly.tools.set_credentials_file(username='MiladPnh', api_key='IGiGl8WUtimCX7gr3UTq')
from sklearn.preprocessing import StandardScaler
import sys
import numpy as np
import scipy as sp
import matplotlib
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report,confusion_matrix
from com.machinelearningnepal.som.online_som import SOM
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.model_selection import train_test_split

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

clf = MLPRegressor(hidden_layer_sizes=(50,50), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001,
                    power_t='0.5', max_iter=5000, shuffle=True, random_state=None, tol=0.0000001, verbose=True, warm_start=False, nesterovs_momentum=True,
                     early_stopping=False, validation_fraction=0.2, beta_1=0.80, beta_2=0.999, epsilon=1e-08)
clf.fit(X_train, y_train)

labels1 = ['train', 'train-learned']
labels2 = ['true', 'prediction']

predictions = clf.predict(X_train)
plt.plot(y_train)
plt.plot(predictions)
plt.legend(labels1)
plt.show()
predictions = clf.predict(X_test)
plt.plot(y_test)
plt.plot(predictions)
plt.legend(labels2)
plt.show()
# print(confusion_matrix(y_test,predictions))
# print(classification_report(y_test,predictions))
print(clf.score(X_test, y_test, sample_weight=None))
print(clf.n_outputs_)
print(clf.coefs_) #    Finding Weights of Hidden Layers
print(clf.intercepts_ ) #  Finding Bias of Hidden Layers

'''
Iteration 1, loss = 0.01514952
Iteration 2, loss = 0.00574804
Iteration 3, loss = 0.00746086
Iteration 4, loss = 0.00654286
Iteration 5, loss = 0.00483146
Iteration 6, loss = 0.00504603
Iteration 7, loss = 0.00563713
Iteration 8, loss = 0.00478175
Iteration 9, loss = 0.00424921
Iteration 10, loss = 0.00417891
Iteration 11, loss = 0.00422573
Iteration 12, loss = 0.00425389
Iteration 13, loss = 0.00419070
Training loss did not improve more than tol=0.000000 for two consecutive epochs. Stopping.
'''



















