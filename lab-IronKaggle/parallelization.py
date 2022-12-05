import time
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def n_neighbours_test(n_neighbour, X_train, X_test, y_train, y_test):
    # create empty model
    knn_c = KNeighborsRegressor(n_neighbors=n_neighbour)

    # training the model
    knn_c.fit(X_train, y_train)

    # evaluate & append TEST
    test_score = knn_c.score(X_test, y_test)

    # evaluate & append TRAIN
    training_score = knn_c.score(X_train, y_train)

    pd.to_pickle(pd.Series([test_score, training_score]), 'knn_scores_' + str(n_neighbour))

def max_depht_test(max_depht, X_train, X_test, y_train, y_test):
    # create empty model
    forest = RandomForestRegressor(
    n_estimators=30,
    max_depth=26
)
    # training the model
    forest.fit(X_train, y_train)