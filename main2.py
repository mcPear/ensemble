from util import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier as CARTClassifier
from sklearn.ensemble import AdaBoostClassifier
from util.validation import k_fold
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

wine = datasets.wine()
pima = datasets.pima()
glass = datasets.glass()
k = 9

datasets = [wine, pima, glass]

dims = [len(Xy[0][0]) for Xy in datasets]
sizes = [len(Xy[0]) for Xy in datasets]


def bag_size(k, percent, size):
    res = size * (k - 1) // k * percent // 100 - 1
    return res


def bag_dim(percent, dim):
    return dim * percent // 100


# # ---single---
# estimator = CARTClassifier()
# f1 = k_fold((X, y), estimator, k, True)
# print(f1)
#
# # --- BAGGING/RF---
# estimator = CARTClassifier()
#
# bagging = BaggingClassifier(estimator, 50, bag_size(k, 60, len(X)), bag_dim(100), n_jobs=4)
# f1 = k_fold((X, y), bagging, k, True)
# print(f1)
#
# # --- boosting ---
# ada = AdaBoostClassifier(estimator, 100)
# f1 = k_fold((X, y), ada, k, True)
# print(f1)

# -------loop------
estimator = CARTClassifier()

shift = 1
size_percs = [x + shift for x in range(100-shift)]
dim_percs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results = np.zeros((len(datasets), len(size_percs)))
for d_i in range(len(datasets)):
    dataset = datasets[d_i]
    for s_i in range(len(size_percs)):
        size_perc = size_percs[s_i]
        ada = AdaBoostClassifier(estimator, size_perc)
        f1 = k_fold(dataset, ada, k, True)
        results[d_i][s_i] = f1

argmaxes = np.zeros(len(datasets))
maxes = np.zeros(len(datasets))
for d_i in range(len(results)):
    print(results[d_i][0])
    argmaxes[d_i] = np.argmax(results[d_i]) + shift
    maxes[d_i] = np.max(results[d_i])

plt.axvline(x=argmaxes[0], color='red', linestyle='dashed')
plt.axvline(x=argmaxes[1], color='green', linestyle='dashed')
plt.axvline(x=argmaxes[2], color='blue', linestyle='dashed')
plt.ylim(top=1, bottom=0)

plt1, = plt.plot(size_percs, results[0], color='red', label='wine')
plt2, = plt.plot(size_percs, results[1], color='green', label='pima')
plt3, = plt.plot(size_percs, results[2], color='blue', label='glass')

legend1 = plt.legend(handles=[plt1, plt2, plt3], loc=1)

plt.gca().add_artist(legend1)

plt.xlabel(f"Liczba słabych klasyfikatorów\noptimum wine = {argmaxes[0]} ({maxes[0]})\noptimum pima = {argmaxes[1]} ({maxes[1]})\noptimum glass = {argmaxes[2]} ({maxes[2]})")
plt.ylabel("F-score")
plt.show()

# todo
# dla baggingu liczba klasyfikatorów/bag_size, sztywne 100 proc atrybutów
# dla rf heatmapa size/dim, na rozmiarze z baggingu może
# dla adabooosta tylko liczba klasyfikatorów i obczaj jakie on rozmiary zakłada

# wine default 0.90221694243
# pima default 0.666727475827
# glass default 0.583346413489

# wine optim 0.905111488608
# pima optim 0.688000439717
# glass optim 0.596197724513

# wine ada 0.91
# pima ada 0.685
# glass ada 0.637