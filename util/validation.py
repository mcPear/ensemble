from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
import numpy as np


def k_fold(data, estimator, k, stratified, scoring='f1_macro', shuffle=False):
    attrs, classes = data
    cv = StratifiedKFold(n_splits=k, shuffle=shuffle) if stratified else KFold(n_splits=k, shuffle=shuffle)
    scores = cross_validate(estimator, attrs, classes, cv=cv, scoring=scoring)
    return np.mean(scores['test_score'])
