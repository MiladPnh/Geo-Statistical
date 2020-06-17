from pandas import read_csv, DataFrame
import numpy as np

filename = "ln_skin_ln_insulin_imp_data.csv"
names = ['preg', 'gluc', 'dbp', 'skin', 'insul', 'bmi', 'pedi', 'age', 'class']
dataset = read_csv(filename, names=names)
# Compute ratio of insulin to glucose
# dataset['ratio'] = dataset['insul']/dataset['gluc']

from __future__ import print_function

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from time import time

from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load

# split dataset into inputs and outputs
print(dataset.head())

values = dataset.values
X = values[:, 0:8]
print(X.shape)
y = values[:, 8]
# print(y.shape)

# def main():

# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
names = ['preg', 'gluc', 'dbp', 'skin', 'insul', 'bmi', 'pedi', 'age']

print("Training GBRT...")
model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, loss='deviance', random_state=1)
t0 = time()
model.fit(X_train, y_train)
print(" done.")

print("done in %0.3fs" % (time() - t0))
importances = model.feature_importances_

print(importances)

# print('Convenience plot with ``partial_dependence_plots``')

print('Custom 3d plot via ``partial_dependence``')
fig = plt.figure()

target_feature = (4, 6)
pdp, axes = partial_dependence(model, target_feature, X=X_train, grid_resolution=50)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(names[target_feature[0]])
ax.set_ylabel(names[target_feature[1]])
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
print(model)

# import dump / load sklearn libs
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load

# import pickle

# save model to disk
filename = 'model.sav'
# pickle.
dump(model, filename)

# load model from disk
interact_insul_pedi = load(filename)
# pickle.load(filename)

# test pdpbox
import pdpbox
from pdpbox import pdp

pdp_pedi_insul = pdp.pdp_interact(interact_insul_pedi, dataset[names], ['insul', 'pedi'])
pdp.pdp_interact_plot(pdp_pedi_insul, ['insul', 'pedi'], center=True, plot_org_pts=True, plot_lines=True,
                      frac_to_plot=0.5)

# pdp.pdp_plot(pdp_diab,'insul')