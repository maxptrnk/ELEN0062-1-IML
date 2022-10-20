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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# (Question 1)

# Put your funtions here
# ...
# Make your experiments here   
datasets = [make_dataset1, make_dataset2]
number_of_samples = 1500
training_sets = 1200
max_depth = [1, 2, 4, 8, None]

number_generations = 5
if __name__ == "__main__":
# 1.1 How decision boudary is affected by complexity
    for i in range(len(datasets)):
        # Data set
        X, y = datasets[i](number_of_samples, random_state = 0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size = training_sets,
            shuffle = False
        )

        for j in range(len(max_depth)):
            # Classifier
            dtc = DecisionTreeClassifier(max_depth = max_depth[j])
            dtc.fit(X_train, y_train)

            # Plot
            plot_boundary(
                "plot\make_dataset" + str(i + 1) + "_depth" + str(max_depth[j]),
                dtc,
                X_test[0:training_sets],
                y_test[0:training_sets],
                title = "make_dataset" + str(i + 1) + "_depth" + str(max_depth[j])
            )
    # 1.2 Accuracies
    print("make_data", "max_depth", "mean", "std")

    for i in range(len(datasets)):
        for j in range(len(max_depth)):
            accr = np.empty(number_generations)
            for k in range(number_generations):
                # Data set
                X, y = datasets[i](number_of_samples, random_state = k)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    train_size = training_sets,
                    shuffle = False
                )
                # Classifier
                dtc = DecisionTreeClassifier(max_depth = max_depth[j])
                dtc.fit(X_train, y_train)
                # Accuracy
                accr[k] = dtc.score(X_test, y_test)

            print(i + 1, max_depth[j], "%f" % np.mean(accr), "%f" % np.std(accr))

# (Question 1)

# Put your funtions here
# ...




    #observe how model complexity impacts the classification boundary
    #build several decision tree models with max_depth values of 
    # 1, 
    # 2, 
    # 4, 
    # 8,
    # and None


    # Answer the following questions in your report.
    # 1. Observe how the decision boundary is affected by tree complexity:

    # (a) illustrate and explain the decision boundary 
    # for each hyperparameter value;

    # (b) discuss when the model is clearly underfitting/overfitting 
    # and detail your evidence for each claim; 
    
    # (c) explain why the model seems more confident 
    # when max_depth is thelargest.


    # 2. Report the average test set accuracies over five generations of the dataset,
    # along with the standard deviations, for each value of n_neighbors. 
    # Briefly comment on them.


    


    