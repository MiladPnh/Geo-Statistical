print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
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

##############################################################################
# Load data
df = pd.read_csv('image21_all_pixels_v2.csv').replace(' ', '')
df.dropna()
df = df.apply(pd.to_numeric)


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
y = df['growth_rate']
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


filename = ['model1.sav', 'model2.sav', 'model3.sav', 'model4.sav', 'model5.sav',
            'model6.sav', 'model7.sav', 'model8.sav', 'model9.sav', 'model10.sav']
filename1 = ['model1.png', 'model2.png', 'model3.png', 'model4.png', 'model5.png',
            'model6.png', 'model7.png', 'model8.png', 'model9.png', 'model10.png']


for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100*i)
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
    accEN1.append(r2_score(y_test,ElasticNet(alpha=0.8, l1_ratio=.75).fit(X_train, y_train).predict(X_test)))

        # Fit regression model

    params = {'n_estimators': 750, 'max_depth': 8,
                  'learning_rate': 0.1, 'loss': 'ls', 'max_features': 'auto'}
    clf = ensemble.GradientBoostingRegressor(subsample=0.7, verbose = 1, **params, warm_start = True, presort='auto')

    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)
    print('Accuracy of the GBM on test set: {:.3f} %'.format(clf.score(X_test, y_test)*100))
    accGBDT.append(clf.score(X_test, y_test))
    accGBDT1.append(clf.score(X_train, y_train))
    importance.append(pd.Series(clf.feature_importances_, features_names[3:]).sort_values(ascending=False))
    # for j in range(30):
    #     features = [j]
    #     pdp, axes = partial_dependence(clf,X_train, features,response_method='auto', percentiles=(0, 1), grid_resolution=100, method='auto')
    #     Partialpdp.append(pdp)
    #     Partialaxs.append(axes)
    #     Partialaxss.append((pd.Series(axes)*(b[j]-a[j])+a[j]).tolist())




    dump(clf, filename[i])

pdnames = ['case1_PD.csv','case2_PD.csv','case3_PD.csv','case4_PD.csv','case5_PD.csv','case6_PD.csv'
           ,'case7_PD.csv','case8_PD.csv','case9_PD.csv','case10_PD.csv','case11_PD.csv','case22_PD.csv',
           'case33_PD.csv','case44_PD.csv','case55_PD.csv','case66_PD.csv','case77_PD.csv','case88_PD.csv'
           ,'case99_PD.csv','case1010_PD.csv']


for i in range(10):
    Partialpdp99 = []
    Partialaxs99 = []
    Partialaxss99 = []

    Partialpdp100 = []
    Partialaxs100 = []
    Partialaxss100 = []
    load_lr_model =load(open(filename[i], 'rb'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100*i)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    for j in range(30):
        features = [j]
        pdp, axes = partial_dependence(load_lr_model,X_train, features,response_method='auto', percentiles=(0.01, 0.99), grid_resolution=1000, method='auto')
        Partialpdp99.append(pdp)
        Partialaxs99.append(axes)
        Partialaxss99.append((pd.Series(axes)*(b[j]-a[j])+a[j]).tolist())
    for j in range(30):
        features = [j]
        pdp, axes = partial_dependence(load_lr_model,X_train, features,response_method='auto', percentiles=(0, 1), grid_resolution=1000, method='auto')
        Partialpdp100.append(pdp)
        Partialaxs100.append(axes)
        Partialaxss100.append((pd.Series(axes)*(b[j]-a[j])+a[j]).tolist())



    with open(pdnames[i], 'w', newline='') as f:
        thewriter = csv.writer(f)
        for ii in range(30):
            thewriter.writerow(Partialpdp99[ii][0][:])

    with open(pdnames[i+10], 'w', newline='') as f:
        thewriter = csv.writer(f)
        for ii in range(30):
            thewriter.writerow(Partialpdp100[ii][0][:])

    print('done')


###########
with open('case1_X.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    for ii in range(30):
        thewriter.writerow(Partialaxss99[ii][0][:])
with open('case11_X.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    for ii in range(30):
        thewriter.writerow(Partialaxss100[ii][0][:])

for i in range(10):
    fig = plt.figure(figsize = (10,5))
    load_lr_model = load(open(filename[i], 'rb'))
    feature_importance = load_lr_model.feature_importances_
    feat_imp = pd.Series(feature_importance, features_names[3:]).sort_values(ascending=False)
    plt.rcParams.update({'font.size': 6})
    feat_imp.plot(kind='bar', title='Importance of Features')
    plt.ylabel('Feature Importance')
    plt.show()
    fig.savefig(filename1[i], format='png', dpi=500)



############


y_load_predit=load_lr_model.predict(X_test)
load_lr_model.score(X_test, y_test)


with open('importancee.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    for i in range(10):
        thewriter.writerow(importance[i])

with open('YY.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(y_test)
    thewriter.writerow(y_load_predit)


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

with open('FN.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(features_names)

y_load_predit=load_lr_model.predict(X_test)

from sklearn.ensemble.partial_dependence import plot_partial_dependence


features = [i for i in range(30)]
features_names = features_names[3:]
fig, axs = plot_partial_dependence(clf, X_train, [(9,11)], percentiles=(0, 1), feature_names=features_names, n_jobs=-1, grid_resolution=100)
plt.show()
# fig.suptitle('Partial dependence plots of pre diabetes on risk factors')

fig = plt.figure()

target_feature = (9, 11)
pdp, axes = partial_dependence(clf,X_train, features,response_method='auto', percentiles=(0, 1), grid_resolution=50, method='auto')
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




acc.plt.boxplot()
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
    plt.rcParams.update({'font.size': 6})
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
df = pd.read_csv('Co.csv').replace(' ', '')
df.dropna()
df = df.apply(pd.to_numeric)


C_mat = df.corr()
fig = plt.figure(figsize = (15,15))
plt.title('Correlation')
plt.rcParams.update({'font.size': 30})
sb.heatmap(C_mat, vmax = 1, square = True)
plt.show()
fig.savefig('feature importance.eps', format='eps', dpi=200)


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


