"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset1, make_dataset2
from plot import plot_boundary
from sklearn.neighbors import KNeighborsClassifier

n_points = 1500
n_neighborsArray = np.array([1, 5, 25, 125, 625, 1200])
RS = np.random.RandomState(seed=42)

average_test_set_accuracies = np.array([])
test_set_accuracies_standard_deviation = np.array([])

for n_neighbors in n_neighborsArray:

    test_set_accuracies = np.zeros(5)

    for i in range (5):

        kNeighbors_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

        [X, y] = make_dataset2(n_points, random_state=RS)
        X_training = X[: 1200, :]
        y_training = y[: 1200]
        X_test = X[1200 :, :]
        y_test = y[1200 :]

        kNeighbors_classifier.fit(X_training, y_training)
        test_set_accuracies[i] = kNeighbors_classifier.score(X_test, y_test)
        if (i==0):
            plot_boundary(str(n_neighbors)+"nn_boundary", kNeighbors_classifier, X, y, mesh_step_size=0.1, title="KNeighbors decision boundary for k = " + str(n_neighbors))

    average = np.average(test_set_accuracies)
    standard_deviation = np.std(test_set_accuracies)
    average_test_set_accuracies = np.append(average_test_set_accuracies, average)
    test_set_accuracies_standard_deviation = np.append(test_set_accuracies_standard_deviation, standard_deviation)

plt.plot(n_neighborsArray, average_test_set_accuracies, 'ro')
plt.plot(n_neighborsArray, test_set_accuracies_standard_deviation, 'bo')

max_y_values = np.array([np.amax(average_test_set_accuracies), np.amax(test_set_accuracies_standard_deviation)])
max_y_value = np.amax(max_y_values)
x_margin = 100
y_margin = max_y_value/10
plt.axis([-x_margin, 1200+x_margin, -y_margin, max_y_value+y_margin])

plt.title("Average and standard deviation of test set accuracies in function of k")
plt.xlabel("k")
plt.ylabel("Test set accuracy")
plt.legend(["average", "standard deviation"], loc ="upper right")

plt.savefig('{}.pdf'.format("knn_test_set_accuracies"), transparent=True)

if __name__ == "__main__":
    pass # Make your experiments here
