"""
Author: Sergi Mas-Pujol
Last update: 10/10/2022

Script to select the best model for a given dataset
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (make_scorer, balanced_accuracy_score, mean_absolute_error, f1_score, zero_one_loss,
                             mean_squared_error, euclidean_distances)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier, Ridge, LogisticRegression


RANDOM_GENERATOR = np.random.default_rng(seed=4)


# TODO: Add to the toolbox


def hyper_parameter_tuning(algorithm_name: str, dict_to_tune: dict, X_train: np.array, y_train: np.array,
                           algorithm_type: str, n_jobs=6):
    # TODO: Refine the code if needed and review the documentation
    """
    GridSearch analysis to select the best algorithm

    Args:
        algorithm_name: Nome of the algorithm to study. Same name as in SKLEARN
        dict_to_tune: contains the model parameters to variates and their values
        X_train: Input samples training dataset
        y_train: Labels training dataset
        algorithm_type: {classification, classification_multilabel_binary, regression}
        n_jobs: Number of jobs used to get the hyperparameters. Use -1 to use all possible resources

    Returns:
        The best parameters for the selected model and the best score obtained
    """

    # The fine de base of the model to test
    if algorithm_name == "MLPClassifier":
        algorithm_base = MLPClassifier(max_iter=100)
    elif algorithm_name == 'RandomForestClassifier':
        algorithm_base = RandomForestClassifier(random_state=0)
    elif algorithm_name == 'AdaBoostClassifier':
        algorithm_base = AdaBoostClassifier(random_state=0)
    elif algorithm_name == 'DecisionTreeClassifier':
        algorithm_base = DecisionTreeClassifier(random_state=0)
    elif algorithm_name == 'KNeighborsClassifier':
        algorithm_base = KNeighborsClassifier(n_jobs=n_jobs)
    elif algorithm_name == 'LinearSVC':
        algorithm_base = LinearSVC(random_state=0, max_iter=200)
    elif algorithm_name == 'GaussianNB':
        algorithm_base = GaussianNB()
    elif algorithm_name == 'NearestCentroid':
        algorithm_base = NearestCentroid()
    elif algorithm_name == 'SGDClassifier':
        algorithm_base = SGDClassifier(max_iter=200, n_jobs=n_jobs, random_state=0)
    elif algorithm_name == 'MLPRegressor':
        algorithm_base = MLPRegressor(max_iter=30, random_state=0)
    elif algorithm_name == 'RandomForestRegressor':
        algorithm_base = RandomForestRegressor(n_jobs=n_jobs, random_state=0)
    elif algorithm_name == 'AdaBoostRegressor':
        algorithm_base = AdaBoostRegressor(random_state=0)
    elif algorithm_name == 'DecisionTreeRegressor':
        algorithm_base = DecisionTreeRegressor(random_state=0)
    elif algorithm_name == 'KNeighborsRegressor':
        algorithm_base = KNeighborsRegressor(n_jobs=n_jobs)
    elif algorithm_name == 'Ridge':
        algorithm_base = Ridge(random_state=0)
    elif algorithm_name == 'LogisticRegression':
        algorithm_base = LogisticRegression(max_iter=50, n_jobs=n_jobs, random_state=0)
    else:
        raise Exception(ValueError, f"Algorithm name {algorithm_name} no defined")

    if algorithm_type == 'classification':
        algorithm_gs = GridSearchCV(algorithm_base, dict_to_tune, scoring=make_scorer(balanced_accuracy_score),
                                    n_jobs=n_jobs)
    elif algorithm_type == 'classification_multilabel_binary':
        # Documentation jaccard_score: https://en.wikipedia.org/wiki/Jaccard_index
        algorithm_gs = GridSearchCV(algorithm_base, dict_to_tune, scoring=make_scorer(zero_one_loss),
                                    n_jobs=n_jobs)
        # algorithm_gs = GridSearchCV(algorithm_base, dict_to_tune, scoring=make_scorer(euclidean_distances),
        #                             n_jobs=n_jobs)
        # algorithm_gs = GridSearchCV(algorithm_base, dict_to_tune, scoring=make_scorer(f1_score),
        #                             n_jobs=n_jobs)
    elif algorithm_type == 'regression':
        algorithm_gs = GridSearchCV(algorithm_base, dict_to_tune, scoring=make_scorer(mean_absolute_error),
                                    cv=10, n_jobs=n_jobs)
    else:
        raise Exception(ValueError, f" algorithm type  {algorithm_name} no defined. Options: classification, "
                                    f"classification_multilabel_binary, or regression.")

    algorithm_gs.fit(X_train, y_train)

    best_params = algorithm_gs.best_params_
    best_score = algorithm_gs.best_score_

    return best_params, best_score
