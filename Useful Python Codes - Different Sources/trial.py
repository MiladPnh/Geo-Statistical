print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as r
import math
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from sklearn.inspection.partial_dependence import plot_partial_dependence
from sklearn.inspection.partial_dependence import partial_dependence
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load
import csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, GridSearchCV #Additional scklearn functions
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import metrics   #Additional scklearn functions
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams


df = pd.read_csv('image21_all_pixels.csv').replace(' ', '')
df = df.apply(pd.to_numeric)
df = df.dropna()
target = 'growth_rate'


features_names = df.columns.values.tolist()
X = df[features_names[1:]]

a = X[:].min()
b = X[:].max()

y = df[target]
y = np.asarray(y)


accGBDT = []
#train score
accGBDT1 = []
importance = []
Partialpdp = []
Partialaxs = []
Partialaxss = []
MeanSE = []
Losss = []

filename = ['mode1.sav', 'mode2.sav', 'mode3.sav', 'mode4.sav', 'mode5.sav',
            'mode6.sav', 'mode7.sav', 'mode8.sav', 'mode9.sav', 'mode10.sav',
            'mode11.sav', 'mode12.sav', 'mode13.sav', 'mode14.sav', 'mode15.sav',
            'mode16.sav', 'mode17.sav', 'mode18.sav', 'mode19.sav', 'mode20.sav',
            'mode2l.sav', 'mode22.sav', 'mode23.sav', 'mode24.sav', 'mode25.sav',
            'mode26.sav', 'mode27.sav', 'mode28.sav', 'mode29.sav', 'mode30.sav',
            'mode3l.sav', 'mode32.sav', 'mode33.sav', 'mode34.sav', 'mode35.sav',
            'mode36.sav', 'mode37.sav', 'mode38.sav', 'mode39.sav', 'mode40.sav',
            'mode4l.sav', 'mode42.sav', 'mode43.sav', 'mode44.sav', 'mode45.sav',
            'mode46.sav', 'mode47.sav', 'mode48.sav', 'mode49.sav', 'mode50.sav',
            'mode5l.sav', 'mode52.sav', 'mode53.sav', 'mode54.sav', 'mode55.sav',
            'mode56.sav', 'mode57.sav', 'mode58.sav', 'mode59.sav', 'mode60.sav',
            'mode6l.sav', 'mode62.sav', 'mode63.sav', 'mode64.sav', 'mode65.sav',
            'mode66.sav', 'mode67.sav', 'mode68.sav', 'mode69.sav', 'mode70.sav',
            'mode7l.sav', 'mode72.sav', 'mode73.sav', 'mode74.sav', 'mode75.sav',
            'mode76.sav', 'mode77.sav', 'mode78.sav', 'mode79.sav', 'mode80.sav',
            'mode8l.sav', 'mode82.sav', 'mode83.sav', 'mode84.sav', 'mode85.sav',
            'mode86.sav', 'mode87.sav', 'mode88.sav', 'mode89.sav', 'mode90.sav',
            'mode9l.sav', 'mode92.sav', 'mode93.sav', 'mode94.sav', 'mode95.sav',
            'mode96.sav', 'mode97.sav', 'mode98.sav', 'mode99.sav', 'mode100.sav']

runs = 10

for i in range(runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=127*i+2)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    clf = GBR(n_estimators = 1000,
          learning_rate= 0.05,
          max_depth=8,
          min_samples_split=20,
          max_features='sqrt',
          subsample=0.80,
          random_state=49, warm_start= True,
          loss='huber', verbose= 1)

    clf.fit(X_train, y_train)
    print("Loss: %.4f" % clf.loss_(y_test,clf.predict(X_test)))
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)
    print('Accuracy of the GBM on test set: {:.3f} %'.format(clf.score(X_test, y_test)*100))
    accGBDT.append(clf.score(X_test, y_test))
    accGBDT1.append(clf.score(X_train, y_train))
    importance.append(pd.Series(clf.feature_importances_, features_names[1:]).sort_values(ascending=False))
    MeanSE.append(mse)
    Losss.append(clf.loss_(y_test, clf.predict(X_test)))
    dump(clf, filename[i])


pdnames = ['PartialD_All.csv']

resolut = 1000

XXX = np.zeros([resolut, len(features_names)-1])
for i in range(len(features_names)-1):
    XXX = np.insert(XXX, [i], np.linspace(df[str(features_names[i+1])].min(),df[str(features_names[i+1])].max(),resolut).reshape(resolut,1), axis=1)

XXX = np.delete(XXX, np.s_[len(features_names)-1:], axis=1)

for i in range(runs):
    Partialpdp99 = []
    Partialaxs99 = []
    Partialaxss99 = []
    load_lr_model =load(open(filename[i], 'rb'))

    for j in range(len(features_names)-1):
        features = [j]
        pdp, axes = partial_dependence(load_lr_model, X_train, features, percentiles=(0.01, 0.99), grid_resolution=resolut, method='auto')
        Partialpdp99.append(pdp)
        Partialaxs99.append(axes)
        Partialaxss99.append((pd.Series(axes)*(b[j]-a[j])+a[j]).tolist())



    with open(pdnames[0], 'a', newline='') as f:
        thewriter = csv.writer(f)
        for ii in range(len(features_names)-1):
            thewriter.writerow(Partialpdp99[ii][0][:])
        f.close()


with open('PartialD_X.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    for ii in range(len(features_names)-1):
        thewriter.writerow(Partialaxss99[ii][0][:])

with open('FeatureImpAll.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    for i in range(runs):
        thewriter.writerow(importance[i])

with open('GBDTacc_test.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(accGBDT)

with open('GBDTacc_train.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(accGBDT1)


with open('Loss.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(Losss)

with open('MSE.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(MeanSE)
