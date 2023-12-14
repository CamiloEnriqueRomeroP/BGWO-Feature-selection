"""Module providing a function printing python version."""
# from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def fitness_function(population, feat_train, feat_val, label_train, label_val):
    """Método de validación de población nula y evaluación de fitness"""
    betha = 0.01
    alpha = 0.99
    if sum(population == 1) == 0:
        fitness = float('inf')
    else:
        error = wrapper(feat_train[:, population == 1],
                       feat_val[:, population == 1], label_train, label_val)
        fitness = alpha * error + betha * (
            np.size(feat_train[:, population == 1])/np.size(feat_train))
    return fitness

def wrapper(xtrain, xvalid, ytrain, yvalid):
    """Método Wrapper"""
    #model = DecisionTreeClassifier()
    model = KNeighborsClassifier(n_neighbors=5)
    ytrain_array = ytrain.flatten()
    model.fit(xtrain, ytrain_array)
    y_pred = model.predict(xvalid)
    acc = accuracy_score(yvalid, y_pred)
    error = 1 - acc
    return error
