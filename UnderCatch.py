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

df = pd.read_csv('MLdataset.csv').replace(' ', '')
df = df.apply(pd.to_numeric)
target = 'Fuchs'
df = df.dropna()
df = df.drop('Legates', 1)
# df = df.drop('SR', 1) #for Fuchs & Legates

features_names = df.columns.values.tolist()
features_namess = features_names[1:]
X = df[features_namess]

a = X[:].min()
b = X[:].max()

y = df[target]
y = np.asarray(y)


# accGBDT = []
# #train score
# accGBDT1 = []
# importance = []
# Partialpdp = []
# Partialaxs = []
# Partialaxss = []
# MeanSE = []
# Losss = []
#
#
#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)

params_Fuchs = {'n_estimators': 3000, 'max_depth': 18, 'min_samples_split': 40,
          'learning_rate': 0.01, 'max_features': 0.7, 'subsample': 0.70, 'loss': 'huber'}
params_Leg = {'n_estimators': 2500, 'max_depth': 20, 'min_samples_split': 45,
          'learning_rate': 0.05, 'max_features': 0.8, 'subsample': 0.75, 'loss': 'huber'}

clf = GBR(**params_Fuchs, warm_start=True, verbose=1)


clf.fit(X_train, y_train)
print("Loss: %.4f" % clf.loss_(y_test, clf.predict(X_test)))
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)
print('Accuracy of the GBM on test set: {:.3f} %'.format(clf.score(X_test, y_test) * 100))
print('Accuracy of the GBM on train set: {:.3f} %'.format(clf.score(X_train, y_train) * 100))


# compute test set deviance
# test_score = np.zeros((params_Fuchs['n_estimators'],), dtype=np.float64)
#
# for i, y_pred in enumerate(clf.staged_predict(X_test)):
#     test_score[i] = clf.loss_(y_test, y_pred)
#
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Deviance')
# plt.plot(np.arange(params_Fuchs['n_estimators']) + 1, clf.train_score_, 'b-',
#          label='Training Set Deviance')
# plt.plot(np.arange(params_Fuchs['n_estimators']) + 1, test_score, 'r-',
#          label='Test Set Deviance')
# plt.legend(loc='upper right')
# plt.xlabel('Boosting Iterations')
# plt.ylabel('Deviance')

# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(features_namess)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

features = [3]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [0]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [1]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [2]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [4]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [5]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [(0, 1)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [(0, 2)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [(1, 2)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [(0, 5)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [(1, 4)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [(1, 5)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()

features = [(0, 6)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [(1, 6)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [(2, 6)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [(5, 6)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [(4, 6)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.01, 0.99), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()


fig = plt.figure()

features = ('AveOccup', 'HouseAge')
pdp, axes = partial_dependence(est, X_train, features=features,
                               grid_resolution=20)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                       cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.suptitle('Partial dependence of house value on median\n'
             'age and average occupancy, with Gradient Boosting')
plt.subplots_adjust(top=0.9)

plt.show()



#
#
#
# #Choose all predictors except target & IDcols
# param_grid = {'n_estimators':range(100,10000,500),
#               'learning_rate': [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
#               'max_depth': range(2, 15, 3),
#               'min_samples_split': [3, 5, 8, 12, 15],
#               'max_features': [1.0, 0.85, 0.75, 0.55, 0.35], ## not possible in our example (only 1 fx)
#               'subsample': [1.0, 0.85, 0.75, 0.5]}
# gsearch1 = GridSearchCV(estimator = GBR(warm_start= True,random_state=49, loss='huber', verbose= 1),
#                                         param_grid = param_grid, iid=False, refit=True,
#                                         n_jobs=-1, cv=10, error_score='raise-deprecating',
#                                         return_train_score = True).fit(X_train,y_train)
#
# print(gsearch1.best_params_, gsearch1.best_score_, gsearch1.best_estimator_)
#
accEN = []
accEN1 = []
#test score
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
            'mode21.sav', 'mode22.sav', 'mode23.sav', 'mode24.sav', 'mode25.sav',
            'mode26.sav', 'mode27.sav', 'mode28.sav', 'mode29.sav', 'mode30.sav']
            # 'mode3l.sav', 'mode32.sav', 'mode33.sav', 'mode34.sav', 'mode35.sav',
            # 'mode36.sav', 'mode37.sav', 'mode38.sav', 'mode39.sav', 'mode40.sav',
            # 'mode4l.sav', 'mode42.sav', 'mode43.sav', 'mode44.sav', 'mode45.sav',
            # 'mode46.sav', 'mode47.sav', 'mode48.sav', 'mode49.sav', 'mode50.sav',
            # 'mode5l.sav', 'mode52.sav', 'mode53.sav', 'mode54.sav', 'mode55.sav',
            # 'mode56.sav', 'mode57.sav', 'mode58.sav', 'mode59.sav', 'mode60.sav',
            # 'mode6l.sav', 'mode62.sav', 'mode63.sav', 'mode64.sav', 'mode65.sav',
            # 'mode66.sav', 'mode67.sav', 'mode68.sav', 'mode69.sav', 'mode70.sav',
            # 'mode7l.sav', 'mode72.sav', 'mode73.sav', 'mode74.sav', 'mode75.sav',
            # 'mode76.sav', 'mode77.sav', 'mode78.sav', 'mode79.sav', 'mode80.sav',
            # 'mode8l.sav', 'mode82.sav', 'mode83.sav', 'mode84.sav', 'mode85.sav',
            # 'mode86.sav', 'mode87.sav', 'mode88.sav', 'mode89.sav', 'mode90.sav',
            # 'mode9l.sav', 'mode92.sav', 'mode93.sav', 'mode94.sav', 'mode95.sav',
            # 'mode96.sav', 'mode97.sav', 'mode98.sav', 'mode99.sav', 'mode100.sav']

n_runs = 30 # should be less than number of files we defined in filenames

for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

        # plt.scatter(np.asarray(X_train[:,17]),y_train)
        # plt.show()

#l1_ratio ==> 1 is Lasso > L1 norm regularization
#alpha ==> Constant that multiplies the penalty terms. Defaults to 1.0, ``alpha = 0`` is equivalent to an ordinary least square.
    accEN.append(r2_score(y_test,ElasticNet(alpha=1, l1_ratio=1).fit(X_train, y_train).predict(X_test)))
    accEN1.append(r2_score(y_test,ElasticNet(alpha=0.85, l1_ratio=.85).fit(X_train, y_train).predict(X_test)))

        # Fit regression model

    clf = GBR(n_estimators = list(gsearch1.best_params_.values())[0],
          learning_rate= list(gsearch1.best_params_.values())[1],
          max_depth=list(gsearch1.best_params_.values())[2],
          min_samples_split=list(gsearch1.best_params_.values())[3],
          max_features=list(gsearch1.best_params_.values())[4],
          subsample=list(gsearch1.best_params_.values())[5],
          random_state=42, warm_start= True,
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

# XXX = np.zeros([resolut, len(features_names)-1])
# for i in range(len(features_names)-1):
#     XXX = np.insert(XXX, [i], np.linspace(df[str(features_names[i+1])].min(),df[str(features_names[i+1])].max(),resolut).reshape(resolut,1), axis=1)
#
# XXX = np.delete(XXX, np.s_[len(features_names)-1:], axis=1)


features = [0, 1, (0, 1)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=2, method='auto')
plt.show()
features = [1, 2, (1, 2)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=2, method='auto')
plt.show()
features = [0, 2, (0, 2)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [1,3, (1, 3)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [0, 4, (0, 4)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [1, 4, (1, 4)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [2, 4, (2, 4)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [0, 5, (0, 5)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [1, 5, (1, 5)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [2, 5, (2, 5)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [0, 6, (0, 6)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [1, 6, (1, 6)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [2, 6, (2, 6)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [5, 6, (5, 6)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()
features = [4, 6, (4, 6)]
plot_partial_dependence(clf,X_train,features, feature_names= features_namess, percentiles=(0.05, 0.95), grid_resolution=1000, n_jobs=-1, method='auto')
plt.show()

for i in range(n_runs):
    Partialpdp99 = []
    Partialaxs99 = []
    Partialaxss99 = []
    load_lr_model =load(open(filename[i], 'rb'))

    for j in range(len(features_names)-1):
        features = [j]
        pdp, axes = partial_dependence(load_lr_model, X_train, features, percentiles=(0.01, 0.99), grid_resolution=resolut, method='auto') # Can replace X_train with XXX defined to define grid ourselves but required to use "brute" method for Partial Dependency
        Partialpdp99.append(pdp)
        Partialaxs99.append(axes)
        Partialaxss99.append((pd.Series(axes)*(b[j]-a[j])+a[j]).tolist())



    with open(pdnames[0], 'a', newline='') as f:
        thewriter = csv.writer(f)
        for ii in range(len(features_names)-1):
            thewriter.writerow(Partialpdp99[ii][0][:])
        f.close()






    #
    # with open(pdnames[0], 'a', newline='') as f:
    #     thewriter = csv.writer(f)
    #     for ii in range(7):
    #         thewriter.writerow(Partialpdp99[ii][0][:])
    #     f.close()

with open('PartialD_X.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    for ii in range(len(features_names)-1):
        thewriter.writerow(Partialaxss99[ii][0][:])

# with open('FeatureImpAll.csv', 'w', newline='') as f:
#     thewriter = csv.writer(f)
#     for i in range(n_runs):
#         thewriter.writerow(importance[i])
#
# with open('GBDTacc_test.csv', 'w', newline='') as f:
#     thewriter = csv.writer(f)
#     thewriter.writerow(accGBDT)
#
# with open('GBDTacc_train.csv', 'w', newline='') as f:
#     thewriter = csv.writer(f)
#     thewriter.writerow(accGBDT1)
#
#
# with open('Loss.csv', 'w', newline='') as f:
#     thewriter = csv.writer(f)
#     thewriter.writerow(Losss)
#
# with open('MSE.csv', 'w', newline='') as f:
#     thewriter = csv.writer(f)
#     thewriter.writerow(MeanSE)
#
# with open('LRacc_Lasso.csv', 'w', newline='') as f:
#     thewriter = csv.writer(f)
#     thewriter.writerow(accEN)
#
# with open('LRacc_EN.csv', 'w', newline='') as f:
#     thewriter = csv.writer(f)
#     thewriter.writerow(accEN1)
#
# with open('FN.csv', 'w', newline='') as f:
#     thewriter = csv.writer(f)
#     thewriter.writerow(features_names)
#
#
#
#
#
# importances = load_lr_model.feature_importances_
# std = np.std([tree.feature_importances_ for tree in load_lr_model],
#              axis=0)
# indices = np.argsort(importances)[::-1]
#
# # Print the feature ranking
# print("Feature ranking:")
#
# for f in range(X.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.show()




# def modelfit(alg, xt, yt, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
#     # Fit the algorithm on the data
#     alg.fit(xt, yt)
#
#     # Predict training set:
#     dtrain_predictions = alg.predict(xt)
#     dtrain_predprob = alg.predict_proba(xt)[:, 1]
#
#     # Perform cross-validation:
#     if performCV:
#         cv_score = cross_val_score(alg, xt, yt, cv=cv_folds,
#                                                     scoring='roc_auc')
#
#     # Print model report:
#     print
#     "\nModel Report"
#     print
#     "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
#
#     if performCV:
#         print
#         "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
#         np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))
#
#     # Print Feature Importance:
#     if printFeatureImportance:
#         feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
#         feat_imp.plot(kind='bar', title='Feature Importances')
#         plt.ylabel('Feature Importance Score')
#
#
# #Choose all predictors except target & IDcols
# predictors = [x for x in df.columns if x not in [target]]
# gbm0 = GradientBoostingClassifier(random_state=10)
# modelfit(gbm0, df, predictors)


import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
df1 = pd.DataFrame(df,columns=['Fuchs','Legates','W','T','RH','Z','E','RR', 'SD', 'SR'])

corrMatrix = df1.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()