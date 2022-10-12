"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin



class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):


    def fit(self, X, y, lda=False):
        """Fit a linear discriminant analysis model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        self.lda = lda


        print("Frist step done")

        # ====================
        # TODO your code here.
        # ====================

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # ====================
        # TODO your code here.
        # ====================

        # return y

        pass

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # TODO your code here.
        # ====================

        # return p

        pass


if __name__ == "__main__":
    #from data import make_data
    from data import make_dataset1
    from plot import plot_boundary

    # generate dataset
    features, labels = make_dataset1(1500,None)

    #Training
    trainingfeatures = features[300:]
    trainingsamples = labels[300:]
    # print(trainingfeatures.shape, trainingsamples.shape)
    trainingset = trainingfeatures,trainingsamples

    # print(trainingset)

    

    #Testing
    testingfeatures = features[:300]
    testingsamples = labels[:300]

    # print(trainingfeatures, trainingsamples)
    # print(trainingfeatures.shape, trainingsamples.shape)

    # print(testingfeatures, testingsamples)
    # print(testingfeatures.shape, testingsamples.shape)

    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X=trainingfeatures, y=trainingsamples)
    #what/where are the targets ??

    #plot_boundary("plot_qda",fitted_estimator, X, y, mesh_step_size=0.1, title="plot_qda")


    
