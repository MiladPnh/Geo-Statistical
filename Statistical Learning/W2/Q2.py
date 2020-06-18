#Dimensionality Reduction

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

import glob
path = 'datahw2/*.csv'
error = np.zeros((10,2))
j = -1
print('Size of different Dataset (sample,feature space)')
for fname in glob.glob(path):
    j = j+1
    data = np.loadtxt(fname, delimiter=',')
    X = data[:,:-1]
    X_orig = X.copy()
    y = data[:,-1]
    print(X.shape)

    # perform PCA
    pca = decomposition.PCA(n_components=3) # 3 features are chosen
    pca.fit(X)
    X = pca.transform(X)

    # Use logistic regression to compare the classification error for when PCA is done and when it is not
    lr = LogisticRegression()

    # Use 5 fold Cross-validation
    # Error for original data
    cv1 = ShuffleSplit(n_splits=5, test_size=0.2)
    scores = cross_val_score(lr,X_orig,y,cv = cv1)
    error[j,0] = 1-scores.mean()
    # Error after PCA
    cv1pca = ShuffleSplit(n_splits=5, test_size=0.2)
    scorespca = cross_val_score(lr,X,y,cv = cv1pca)
    error[j,1] = 1-scorespca.mean()
print('\n Error Matrix:')
print(error)

######################################################################
#Output
#Size of different Dataset (sample,feature space)
# (4177, 8)
# (120, 6)
# (120, 6)
# (16281, 14)
# (32561, 14)
# (100, 31)
# (798, 31)
# (452, 262)
# (25, 59)
# (194, 59)
#
#  Error Matrix:
# [[ 0.34736842  0.4277512 ]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [ 0.16112987  0.19023641]
#  [ 0.15882082  0.18968217]
#  [ 0.29        0.25      ]
#  [ 0.14625     0.18375   ]
#  [ 0.31648352  0.3956044 ]
#  [ 0.08        0.28      ]
#[ 0.19487179  0.71282051]]


myfile = open('pca_error_list.txt', mode='wt', encoding='utf-8')
for j in range(len(error)):
    myfile.write(','.join(map(str,np.ndarray.tolist(error[j])))+'\n')
myfile.close()

# Feeding the calculated errors to the Friedman's test
import re
from math import floor
f = open('pca_error_list.txt','r') # Reading the text file:
                               # Note that the text file must have a new line character at the very end
def truncate(fl, n):
    return floor(fl * 10 ** n) / 10 ** n
classifierrank = [0]*6
N = 0
for m in f:
    N = N+1
    match = re.findall('([\w.]+),',m)
    match.append(re.findall('([\w.]+)\n',m)[0])
    matchnum = [truncate(float(match[num]),2) for num in range(len(match))]
    if len(matchnum)!=2:
        yoyo
    matchsort = sorted(matchnum,reverse = True)
    matchrank = [[matchsort[a],a+1] for a in range(len(matchnum))]
    for a in matchnum:      # rank Assignment for each classifier
        if matchnum.count(a)>1:
            rankequal = [enum+1 for enum in range(len(matchnum)) if matchsort[enum]==a]
            for b in range(len(rankequal)):
                matchrank[rankequal[b]-1][1] = float(sum(rankequal))/len(rankequal)
    indexsort = sorted(range(len(matchnum)), key=lambda k: matchnum[k],reverse = True)
    for a in range(len(matchnum)):      # Calculating average rank of each classifier
        for b in range(len(indexsort)):
            if indexsort[b]==a:
                classifierrank[a] = classifierrank[a]+matchrank[b][1]
# print(matchnum)
# print(matchrank)
# print(matchsort)
# print(indexsort)
classifiersumrank = [float(a)/N for a in classifierrank]       # The average rank vector
# print(classifierrank)
# print(classifiersumrank)
sumsq = sum(map(lambda x:x*x,classifiersumrank))
chi2f = float(12*N)/(2*3)*(sumsq-float(2*9)/4)
print("\n\nThe Friedman's score is", chi2f, '\n\n')
differencemat = [[0]*len(classifiersumrank)]*len(classifiersumrank)
# print(differencemat[2][:])
for a in range(len(classifiersumrank)):
    differencemat[a] = [max(classifiersumrank[a]-b,0) for b in classifiersumrank]
    print(differencemat[a])                                        # Generating the realtive difference matrix



#####################################Output#####################
# The Friedman's score is 3.5999999999999943
#
#
# [0.0, 0.6000000000000001, 1.8, 1.8, 1.8, 1.8]
# [0, 0.0, 1.2, 1.2, 1.2, 1.2]
# [0, 0, 0.0, 0.0, 0.0, 0.0]
# [0, 0, 0.0, 0.0, 0.0, 0.0]
# [0, 0, 0.0, 0.0, 0.0, 0.0]
# [0, 0, 0.0, 0.0, 0.0, 0.0]







