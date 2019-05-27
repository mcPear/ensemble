import csv, random
from util import utils
from sklearn.datasets import load_iris
import numpy as np


def pima():
    return load('data/diabetes_data')


def glass():
    return load('data/glass_data')


def wine():
    return load('data/wine_data')


def iris():
    iris = load_iris()
    X, y = [[elem for elem in record] for record in iris.data], np.asarray(iris.target)
    return [X, y]


def load(file):
    iterator = csv.reader(open(file, "r"))
    data = list(iterator)
    random.shuffle(data)
    class_index = utils.get_class_index(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            val = data[i][j]
            data[i][j] = float(val) if j != class_index else val
    return utils.horizontal_split(data)
