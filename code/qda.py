"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
#from code.plot import plot_with_colors

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

        # ====================
        self.priors = dict()
        self.means = dict()
        self.covs = dict()
        self.classes = np.unique(y) #[0. 1.]

        for c in self.classes: # pour chaque classe
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0] # on calcule de la prior des 2 classes : prob de la class sur le dataset
            self.means[c] = np.mean(X_c, axis=0) # moyen des 2 classes

            if lda:
                self.covs[c] = np.cov(X, rowvar=False) # matrice de covariance du dataset
            else:
                self.covs[c] = np.cov(X_c, rowvar=False) # matrice de covariance des 2 classes

        return self
        # ====================


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
        return self.predict_proba(X).argmax(axis=1)
        # ====================

   
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
        prob_X = []

        for x in X:
            prob_x = []
            for c in self.classes:
                numerator = self.f(x, self.means[c], self.covs[c]) * self.priors[c]
                denominator = 0
                for c2 in self.classes:
                    denominator += self.f(x, self.means[c2], self.covs[c2]) * self.priors[c2]
                prob_x.append(round (numerator / denominator,3))
            prob_X.append(prob_x)

        return np.array(prob_X)
        # ====================

    def f(self, x, μ, Σ):
            """
            The density function of multivariate normal distribution.

            Parameters
            ---------------
            x: ndarray(float, dim=2)
                random vector, N by 1
            μ: ndarray(float, dim=1 or 2)
                the mean of x, N by 1
            Σ: ndarray(float, dim=2)
                the covarianece matrix of x, N by 1
            """

            N = x.size

            temp1 = np.linalg.det(Σ) ** (-1/2)
            temp2 = np.exp(-.5 * (x - μ).T @ np.linalg.inv(Σ) @ (x - μ))

            return ( 1/( ((2 * np.pi) ** (N/2)) * temp1) ) * temp2



if __name__ == "__main__":
    #from data import make_data
    from plot import plot_boundary
    from data import make_dataset1

    # generate dataset
    features, labels = make_dataset1(1500,None)
    trainingfeatures = features[300:]
    trainingtragets = labels[300:]
    testingfeatures = features[:300]
    testingtargets = labels[:300]

    print("QDA")
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(trainingfeatures, trainingtragets, lda=False)
    #qda.fit(trainingfeatures, trainingtragets, lda=True)        
    proba_per_classes = qda.predict_proba(testingfeatures)
    prediction_classes = qda.predict(testingfeatures)

    plot_boundary("test1223",qda,testingfeatures, testingtargets)

    #verify the model 
    #look at max proba_per_classes and compare to the real class of the sample 



    



    
