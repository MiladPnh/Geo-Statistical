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
import seaborn as sb
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
##############################################################################
# Load data
df = pd.read_csv('image21_all_pixels_v2.csv').replace(' ', '')
df = df.apply(pd.to_numeric)
df.dropna()


# def encode_and_bind(original_dataframe, feature_to_encode):
#     dummies = pd.get_dummies(original_dataframe[feature_to_encode])
#     res = pd.concat([original_dataframe, dummies], axis=1)
#     return(res)
#
# df = encode_and_bind(encode_and_bind(df, 'lat'),'lon')

features_names = df.columns.values.tolist()
X = df[features_names[3:]]

a = X[:].min()
b = X[:].max()

# labelencoder = LabelEncoder()
# X.iloc[:,0] = labelencoder.fit_transform(X.iloc[:,0].astype(str).str[:])
# X.iloc[:,1] = labelencoder.fit_transform(X.iloc[:,1].astype(str).str[:])
# onehotencoder = OneHotEncoder(categorical_features = [0,1])
# X = onehotencoder.fit_transform(X).toarray()
# onehotencoder = OneHotEncoder(categorical_features = [43])
# X = onehotencoder.fit_transform(X).toarray()
# to_drop = features_names[33:].append('growth_rate')
# X = df.drop(columns=[range(33:)], inplace=True, axis=1)


#X = df.drop('Class', axis=1)
# X = df.drop('growth_rate', axis=1)
y = df[features_names[0]]
y = np.asarray(y)

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


for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
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

    params = {'n_estimators': 400, 'max_depth': 8,
                  'learning_rate': 0.002, 'loss': 'huber', 'max_features': 0.5}
    clf = GBR(subsample=0.85, verbose = 1, **params, warm_start = True, presort='auto')

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


# from sklearn import linear_model
# clfL = linear_model.LinearRegression()
# clfL.fit(X, y)
#
# y_pred = clfL.predict(X_test)
# print('Coefficients: \n', clfL.coef_)
# print("Mean squared error: %.2f"
#       % mean_squared_error(y_test, y_pred))
# print('Variance score: %.2f' % r2_score(y_test, y_pred))
# clfL.score(X_test, y_test)
#
# plt.scatter(X_test, y_test,  color='black')
# plt.plot(X_test, y_pred, color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())
# plt.show()

T_Avg = np.linspace(284.4390623, 297.1991269, 1000).reshape(1000,1)
WindHorSpeed_Avg = np.linspace(1.831389237, 15.10089904, 1000).reshape(1000,1)
WindDir_Avg = np.linspace(6.232211055, 352.7591878, 1000).reshape(1000,1)
WindVerSpeed_Avg = np.linspace(0.01307508, 0.279437024, 1000).reshape(1000,1)
TKE_Avg = np.linspace(0.0001, 0.279961984, 1000).reshape(1000,1)
Inv_Base_Height = np.linspace(66.73569218, 820.0051593, 1000).reshape(1000,1)
Rain_Rate_Avg = np.linspace(0.0001, 6.310261377, 1000).reshape(1000,1)

XXX = np.insert(XXX, [6], Rain_Rate_Avg, axis=1)

a = [284.4390623, 1.831389237, 6.232211055, 0.01307508, 0.0001, 66.73569218, 0.0001]
b = [297.1991269, 15.10089904, 352.7591878, 0.279437024, 0.279961984, 820.0051593, 6.310261377]
for i in range(100):
    Partialpdp99 = []
    Partialaxs99 = []
    Partialaxss99 = []
    load_lr_model =load(open(filename[i], 'rb'))
    # a = XXX[:].min()
    # b = XXX[:].max()
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=78*i)
    # scaler = MinMaxScaler()
    # scaler.fit(XXX)
    # XXX = scaler.transform(XXX)
    for j in range(7):
        features = [j]
        pdp, axes = partial_dependence(load_lr_model, XXX, features, percentiles=(0.01, 0.99), grid_resolution=1000, method='brute')
        Partialpdp99.append(pdp)
        Partialaxs99.append(axes)
        Partialaxss99.append((pd.Series(axes)*(b[j]-a[j])+a[j]).tolist())



    with open(pdnames[0], 'a', newline='') as f:
        thewriter = csv.writer(f)
        for ii in range(7):
            thewriter.writerow(Partialpdp99[ii][0][:])
        f.close()


with open('PartialD_X.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    for ii in range(7):
        thewriter.writerow(Partialaxss99[ii][0][:])

with open('FeatureImpAll.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    for i in range(100):
        thewriter.writerow(importance[i])

with open('GBDTacc_test.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(accGBDT)

with open('GBDTacc_train.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(accGBDT1)

with open('LRacc_Lasso.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(accEN)

with open('LRacc_EN.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(accEN1)


with open('Loss.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(Losss)

with open('MSE.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(MeanSE)

with open('FN.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(features_names)

for i in range(10):
    fig = plt.figure(figsize = (10,5))
    #load_lr_model = load(open(filename[i], 'rb'))
    # feature_importance = clf.feature_importances_
    feat_imp = pd.Series(importance[i], features_names[1:]).sort_values(ascending=False)
    plt.rcParams.update({'font.size': 14})
    feat_imp.plot(kind='bar', title='Importance of Features')
    # plt.ylabel('Feature Importance')
    plt.show()
    fig.savefig(filename1[i], format='png', dpi=500)

plt.scatter(y_test,clf.predict(X_test))
plt.ylabel('predict')
plt.xlabel('observation')
plt.show()
############


y_load_predit=load_lr_model.predict(X_test)
load_lr_model.score(X_test, y_test)



with open('YY.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(y_test)
    thewriter.writerow(y_load_predit)



y_load_predit=load_lr_model.predict(X_test)

from sklearn.ensemble.partial_dependence import plot_partial_dependence

features = [i for i in range(30)]
##
pdp, axes = partial_dependence(clf,X_train, [2],response_method='auto', percentiles=(0.01, 0.99), grid_resolution=1000, method='auto')
fig, axs = plot_partial_dependence(clf, X_train, [1,6,6], percentiles=(0, 1), feature_names=features_names, n_jobs=-1, grid_resolution=100)
plt.show()
# fig.suptitle('Partial dependence plots of pre diabetes on risk factors')
##
fig = plt.figure()

target_feature = (6, 5)
pdp, axes = partial_dependence(clf,X_train, target_feature,response_method='auto', percentiles=(0, 1), grid_resolution=50, method='auto')
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(features_names[target_feature[0]+1])
ax.set_ylabel(features_names[target_feature[1]+1])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=142)
plt.colorbar(surf)
plt.suptitle('Partial dependence of pre diabetes risk factors')

plt.subplots_adjust(right=1, top=.9)

plt.show()


plt.subplots_adjust(bottom=0.1, right=1.1, top=1.4)  # tight_layout causes overlap with suptitle



import pdpbox
from pdpbox import pdp
pdp_pedi_insul = pdp.pdp_interact(interact_insul_pedi,dataset[names],['insul','pedi'])
pdp.pdp_interact_plot(pdp_pedi_insul, ['insul','pedi'], center=True, plot_org_pts=True, plot_lines=True, frac_to_plot=0.5)


# features = [0]
# fig, axs = plot_partial_dependence(clf, X_train[:,45:], features, feature_names=features_names[3:], n_jobs=-1, grid_resolution=50)
# # fig.suptitle('Partial dependence plots of pre diabetes on risk factors')
#
# plt.subplots_adjust(bottom=0.1, right=1.1, top=1.4)
# plt.show()

# fig = plt.figure()
#
# target_feature = (15, 13)
# pdp, axes = partial_dependence(clf, target_feature, X=X_train, grid_resolution=50)
# XX, YY = np.meshgrid(axes[0], axes[1])
# Z = pdp[0].reshape(list(map(np.size, axes))).T
# ax = Axes3D(fig)
# surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor='k')
# ax.set_xlabel(features_names[target_feature[1]])
# ax.set_ylabel(features_names[target_feature[0]])
# ax.set_zlabel('Partial dependence')
#     #  pretty init view
# ax.view_init(elev=22, azim=142)
# plt.colorbar(surf)
# plt.suptitle('Partial dependence of pre diabetes risk factors')
#
# plt.subplots_adjust(right=1, top=.9)

plt.show()
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(features_names[target_feature[0]])
ax.set_ylabel(features_names[target_feature[1]])
ax.set_zlabel('Partial dependence')
    #  pretty init view
ax.view_init(elev=22, azim=142)
plt.colorbar(surf)
plt.suptitle('Partial dependence of pre diabetes risk factors')

plt.subplots_adjust(right=1, top=.9)

plt.show()

    # Needed on Windows because plot_partial_dependence uses multiprocessing
    # if __name__ == '__main__':
    #    main()

    # check model
print(clf)




plt.boxplot(accGBDT)
plt.show()
# Plot training deviance


# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
plt.show()

# #############################################################################
for i in range(10):
    fig = plt.figure(figsize = (10,5))
    load_lr_model = load(open(filename1[i], 'rb'))
    feature_importance = load_lr_model.feature_importances_
    feat_imp = pd.Series(feature_importance, features_names[3:]).sort_values(ascending=False)
    plt.rcParams.update({'font.size': 18})
    feat_imp.plot(kind='bar', title='Importance of Features')
    plt.ylabel('Feature Importance')
    plt.show()
    fig.savefig(filename[i], format='png', dpi=500)

#Plot Results
labels1 = ['train', 'train-learn']
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


####################Correlations############
df = pd.read_csv('FinalCor.csv').replace(' ', '')
df.dropna()
df = df.apply(pd.to_numeric)

C_mat = df.corr()
fig = plt.figure(figsize = (25,20))
#sb.heatmap(C_mat, vmin=0, vmax=1, cmap="YlGnBu")
plt.rcParams.update({'font.size': 26})
sb.heatmap(C_mat,vmin=-0.25, vmax=1)
plt.rcParams["font.serif"] = 'Times New Roman'
plt.show()
fig.savefig('Basic6.png', dpi=1000)

#plt.rcParams["font.weight"] = "bold"



correlations = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-10, vmax=10)
fig.colorbar(cax)
names = ['']+list(df)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


#######Parameter Tuning#####
'''
Tuning tree-specific parameters
Tune max_depth and num_samples_split
Tune min_samples_leaf
Tune max_features
'''

#Fix learning rate and number of estimators for tuning tree-based parameters

predictors = [x for x in X_train.columns if x not in [target, IDcol]]
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10),
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10),
param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,max_features='sqrt', subsample=0.8, random_state=10),
param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


################Regression Model#########
LR_baseline = LinearRegression()
LR_baseline.fit(X=X_train, y=y_train)
r_squared_LR_baseline = LR_baseline.score(X=X_test, y=y_test)
print ('R^2(LR_baseline):', r_squared_LR_baseline)




n_estimators = len(clf.estimators_)
def deviance_plot(clf, X_test, y_test, ax=None, label='', train_color='#2c7bb6', test_color='#d7191c', alpha=1.0):
    test_dev = np.empty(n_estimators)
    for i, pred in enumerate(clf.staged_predict(X_test)):
        test_dev[i] = clf.loss_(y_test, pred)
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color, label='Test %s' % label,
        linewidth=2, alpha=alpha)
        ax.plot(np.arange(n_estimators) + 1, clf.train_score_, color=train_color,
        label='Train %s' % label, linewidth=2, alpha=alpha)
        ax.set_ylabel('Error')
        ax.set_xlabel('n_estimators')
        return test_dev, ax

test_dev, ax = deviance_plot(clf, X_test, y_test)
ax.legend(loc='upper right')
# add some annotations
plt.show()


def fmt_params(params):
    return ", ".join("{0}={1}".format(key, val) for key, val in params.iteritems())

for params, (test_color, train_color) in [({}, ('#d7191c', '#2c7bb6')),
({'min_samples_leaf': 3},
('#fdae61', '#abd9e9'))]:
    est = GBR(n_estimators=n_estimators, max_depth=1, learning_rate=1.0)
    est.set_params(**params)
    est.fit(X_train, y_train)

test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params),
train_color=train_color, test_color=test_color)


ax.annotate('Higher bias', xy=(900, est.train_score_[899]), xycoords='data',
 xytext=(600, 0.3), textcoords='data',
 arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
 )
 ax.annotate('Lower variance', xy=(900, test_dev[899]), xycoords='data',
 xytext=(600, 0.4), textcoords='data',
 arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
 )
 plt.legend(loc='upper right')