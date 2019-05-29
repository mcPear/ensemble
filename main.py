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
glass = datasets.glass()
pima = datasets.pima()
k = 9

(X, y) = wine  # specify dataset here
dim = len(X[0])
size = len(X)


def bag_size(k, percent, size):
    res = size * (k - 1) // k * percent // 100 - 1
    return res


def bag_dim(percent):
    return dim * percent // 100


# ---single---
estimator = CARTClassifier()
f1 = k_fold((X, y), estimator, k, True)
print(f1)

# --- BAGGING/RF---
estimator = CARTClassifier()

bagging = BaggingClassifier(estimator, 50, bag_size(k, 60, len(X)), bag_dim(100), n_jobs=4)
f1 = k_fold((X, y), bagging, k, True)
print(f1)

# --- boosting ---
ada = AdaBoostClassifier(estimator, 100)
f1 = k_fold((X, y), ada, k, True)
print(f1)

# -------loop------
size_percs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
dim_percs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results = np.zeros((len(size_percs), len(dim_percs)))
for s_i in range(len(size_percs)):
    size_perc = size_percs[s_i]
    for d_i in range(len(dim_percs)):
        dim_perc = dim_percs[d_i]
        # bagging = BaggingClassifier(estimator, 10, bag_size(k, size_perc, size), bag_dim(100), n_jobs=4)
        random_forest = BaggingClassifier(estimator, 50, bag_size(k, size_perc, size), bag_dim(dim_perc), n_jobs=4)
        f1 = k_fold((X, y), random_forest, k, True)
        results[s_i][d_i] = f1

print(results)

df = pd.DataFrame(results, dim_percs, size_percs)
sns.heatmap(df, annot=True)
plt.xlabel("Bag size - percent of training dataset size")
plt.ylabel("Attributes count - percent of all attributes count")
plt.show()

# todo
# cart - 3 pierwsze liczbowe przebadaj
# klasyfikatory może badaj na domyślnych, a porównaj z optymalnym
# dla baggingu liczba klasyfikatorów/bag_size, sztywne 100 proc atrybutów
# dla rf heatmapa size/dim, na rozmiarze z baggingu może
# dla adabooosta tylko liczba klasyfikatorów i obczaj jakie on rozmiary zakłada
