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
from sklearn.tree import DecisionTreeClassifier


# (Question 1)

# Put your funtions here
# ...


if __name__ == "__main__":
    #pass 
    
    
    DecisionTreeClassifier(make_dataset2)

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


    


    