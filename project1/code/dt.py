"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset1,make_dataset2
from plot import plot_boundary
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

datasets = [make_dataset1,make_dataset2]
dataset = make_dataset2
number_of_samples = 1500
training_sets = 1200
max_depth = [1, 2, 4, 8, None]
number_generations = 5

if __name__ == "__main__":
    # 1.1 Plots to see how decision boudary is affected by complexity
    X, y = dataset(number_of_samples, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = training_sets,shuffle = False)

    for j in range(len(max_depth)):
        # Decision Tree classifier
        dtc = DecisionTreeClassifier(max_depth = max_depth[j])
        dtc.fit(X_train, y_train)
        # printing the Plot
        plot_boundary( "plot\make_dataset2" + "_max_depth" + str(max_depth[j]),dtc, X_test[0:training_sets],y_test[0:training_sets],title = "max_depth : " + str(max_depth[j]))
    # 1.2  
    print("max_depth", "mean", "std")
    for i in range(len(max_depth)):
        accr = np.empty(number_generations)
        for k in range(number_generations):
            # Data set
            X, y = dataset(number_of_samples, random_state = k)
            X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = training_sets,shuffle = False)

            dtc = DecisionTreeClassifier(max_depth = max_depth[i])
            dtc.fit(X_train, y_train)
            accr[k] = dtc.score(X_test, y_test)
        print(max_depth[i],"{:.5f}".format(np.mean(accr)), "{:.5f}".format(np.std(accr)))

    #tuning the hyperparameter of max_depth:
    print("dataset","max_depth", "mean", "std")
    for i in range(len(datasets)):
        for j in range(len(max_depth)):
            accr = np.empty(number_generations)
            for k in range(number_generations):
                # Data set
                X, y = datasets[i](number_of_samples, random_state = k)
                X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = training_sets,shuffle = False)
                dtc = DecisionTreeClassifier(max_depth = max_depth[j])
                dtc.fit(X_train, y_train)
                accr[k] = dtc.score(X_test, y_test)
            print(str(i+1),max_depth[j],"{:.5f}".format(np.mean(accr)), "{:.5f}".format(np.std(accr)))
    for i in range(len(datasets)):
        for k in range(number_generations):
            X, y = datasets[i](number_of_samples, random_state = k)
            X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = training_sets,shuffle = False)
            grid_params = { 'max_depth' : [1, 2, 4, 8, None]}
            gs = GridSearchCV(DecisionTreeClassifier(), grid_params, cv=10)
            g_res = gs.fit(X_train, y_train)
            print("k=", k,"n= ",i)
            print("Best parameter: max_depth =", g_res.best_params_["max_depth"])
            print("Best parameter accuracy:", g_res.best_score_)
    
    score=[[],[]]
    max_depth=[8,4]
    for i in range(len(datasets)):
        
        for k in range(number_generations):
            X, y = datasets[i](number_of_samples, random_state = k)
            X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = training_sets,shuffle = False)
            dtc = DecisionTreeClassifier(max_depth = max_depth[i])
            dtc.fit(X_train,y_train)
            y_pred = dtc.predict(X_test)
            accuracy= accuracy_score(y_pred, y_test)
            score[i].append(accuracy)
    print("__________________________________________________________")
    print(score[0])
    print(score[1])
    print("__________________________________________________________")
    print("dataset", "average accuracy", "average accuracy standard deviation")
    print("1","{:.4f}".format(np.mean(score[0])),"{:.4f}".format(np.std(score[0])))
    print("2","{:.4f}".format(np.mean(score[1])),"{:.4f}".format(np.std(score[0])))
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


    


    