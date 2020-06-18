# Comparing Classifires
import re
from math import floor

# Reading the text file
f = open('hw1-scores.txt','r')

def truncate(fl, n):
    return floor(fl * 10 ** n) / 10 ** n

classifierrank = [0]*6
N = 0
for m in f:
    N = N + 1
    match = re.findall('([\\w.]+),', m)
    match.append(re.findall('([\\w.]+)\\n', m)[0])
    matchnum = [truncate(float(match[num]), 2) for num in range(len(match))]
    if len(matchnum) != 6:
        break
    matchsort = sorted(matchnum, reverse=True)
    matchrank = [[matchsort[a], a + 1] for a in range(len(matchnum))]
    for a in matchnum:  # rank Assignment for each classifier
        if matchnum.count(a) > 1:
            rankequal = [enum + 1 for enum in range(len(matchnum)) if matchsort[enum] == a]
            for b in range(len(rankequal)):
                matchrank[rankequal[b] - 1][1] = float(sum(rankequal)) / len(rankequal)
    indexsort = sorted(range(len(matchnum)), key=lambda k: matchnum[k], reverse=True)
    for a in range(len(matchnum)):  # Calculating average rank of each classifier
        for b in range(len(indexsort)):
            if indexsort[b] == a:
                classifierrank[a] = classifierrank[a] + matchrank[b][1]

classifiersumrank = [float(a)/N for a in classifierrank]       # The average rank vector
sumsq = sum(map(lambda x:x*x,classifiersumrank))
chi2f = float(12*N)/(6*7)*(sumsq-float(6*49)/4)
print("\\n\\n The Friedman's score is", chi2f, '\\n\\n')
differencemat = [[0]*len(classifiersumrank)]*len(classifiersumrank)

for a in range(len(classifiersumrank)):
    differencemat[a] = [max(classifiersumrank[a] - b, 0) for b in classifiersumrank]
    print(differencemat[a])  # Generating the realtive difference matrix