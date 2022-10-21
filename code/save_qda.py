"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

from sre_compile import isstring
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

    def compute_accuracy(self, prediction_classes,testingtargets):

        assert len(prediction_classes) == len(testingtargets)
        well_predicted_classes = 0

        for sample in range(len(testingtargets)): 
            if prediction_classes[sample] == testingtargets[sample]:
                well_predicted_classes += 1

        accuracy = well_predicted_classes/len(testingtargets)
        return accuracy

    def qst():
        # generate dataset
        features, labels = make_dataset2(1500,None) 

        trainingfeatures = features[300:]
        trainingtragets = labels[300:]
        testingfeatures = features[:300]
        testingtargets = labels[:300]

        print("QDA")
        qda = QuadraticDiscriminantAnalysis()

        qda.fit(trainingfeatures, trainingtragets, lda=True)      
        plot_boundary("3.2_lda_data2",qda,testingfeatures, testingtargets, title="3.2_lda_data2")
        # proba_per_classes = qda.predict_proba(testingfeatures)
        # prediction_classes = qda.predict(testingfeatures)

        if len(prediction_classes) == len(testingtargets2): print("prediction_classes && testingtargets2 same LENNNNNNNNNNNNNNNNN")
            # for sample in len(testingtargets2): 
            #     if prediction_classes == testingtargets2:
            #         well_predicted_classes+1
        
    
        qda.fit(trainingfeatures, trainingtragets, lda=False)
        plot_boundary("3.2_qda_data2",qda,testingfeatures, testingtargets, title="3.2_qda_data2")
        # proba_per_classes = qda.predict_proba(testingfeatures)
        # prediction_classes = qda.predict(testingfeatures)


    def question_3_3():


        # transfrom in accuracy, std in dict

        accuracy_dataset1_qda_tmp = None 
        std_deviation_dataset1_qda_tmp = None

        accuracy_dataset1_lda_tmp = None 
        std_deviation_dataset1_lda_tmp = None

        accuracy_dataset2_qda_tmp = None 
        std_deviation_dataset2_qda_tmp = None

        accuracy_dataset2_lda_tmp = None 
        std_deviation_dataset2_lda_tmp = None


        for generation in range(5):

            rd = (generation+10)**4-111
            #print(rd)

            # datasets
            features2, labels2 = make_dataset2(1500,rd) 
            trainingfeatures2 = features2[300:]
            trainingtragets2 = labels2[300:]
            testingfeatures2 = features2[:300]
            testingtargets2 = labels2[:300]

            features1, labels1 = make_dataset1(1500,rd) 
            trainingfeatures1 = features1[300:]
            trainingtragets1 = labels1[300:]
            testingfeatures1 = features1[:300]
            testingtargets1 = labels1[:300]
            
            ## methods
            #QDA-DT1
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(trainingfeatures1, trainingtragets1, lda=False)
            proba_per_classes = qda.predict_proba(testingfeatures1)
            prediction_classes = qda.predict(testingfeatures1)

            #accuracy_dataset1_qda_tmp += 
            #std_deviation_dataset1_qda_tmp +=

            name =  "---qda"+str(generation)
            plot_boundary(name,qda,testingfeatures1, testingtargets1, title=name)

            #QDA-DT2
            qda.fit(trainingfeatures2, trainingtragets2, lda=False)
            name =  "---qda"+str(generation)
            plot_boundary(name,qda,testingfeatures2, testingtargets2, title=name)
            proba_per_classes = qda.predict_proba(testingfeatures2)
            prediction_classes = qda.predict(testingfeatures2)

            



            #LDA-DT1
            qda.fit(trainingfeatures1, trainingtragets1, lda=True)
            proba_per_classes = qda.predict_proba(testingfeatures1)
            prediction_classes = qda.predict(testingfeatures1)
            name =  "---lda"+str(generation)
            plot_boundary(name,qda,testingfeatures1, testingtargets1, title=name)

            #LDA-DT2
            qda.fit(trainingfeatures2, trainingtragets2, lda=True)
            proba_per_classes = qda.predict_proba(testingfeatures2)
            prediction_classes = qda.predict(testingfeatures2)
            name =  "---lda"+str(generation)
            plot_boundary(name,qda,testingfeatures2, testingtargets2, title=name)



        avg_accuracy_dataset1_qda = accuracy_dataset1_qda_tmp / 5 #repeat for others
        std_deviation_dataset1_qda = std_deviation_dataset1_qda_tmp / 5 #repeat for others




def test_method(trainingfeatures,trainingtragets,testingfeatures,testingtargets,fname=None, lda=bool):
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(trainingfeatures, trainingtragets, lda)
            #proba_per_classes = qda.predict_proba(testingfeatures)
            prediction_classes = qda.predict(testingfeatures)

            accuracy = qda.compute_accuracy(prediction_classes,testingtargets)
            std_deviation = np.std(prediction_classes) # not sure about it

            if fname!=None:
                if fname.isstring:
                    plot_boundary(fname,qda,testingfeatures1, testingtargets1, title=fname)

            return accuracy,std_deviation




if __name__ == "__main__":
    #from data import make_data
    from plot import plot_boundary
    from data import make_dataset1, make_dataset2

    
    # transfrom in accuracy, std in dict


    accuracy_dataset1_qda_tmp = 0 
    std_deviation_dataset1_qda_tmp = 0

    accuracy_dataset1_lda_tmp = 0 
    std_deviation_dataset1_lda_tmp = 0

    accuracy_dataset2_qda_tmp = 0 
    std_deviation_dataset2_qda_tmp = 0

    accuracy_dataset2_lda_tmp = 0 
    std_deviation_dataset2_lda_tmp = 0


    for generation in range(5):

        rd = (generation+10)**4-111
        #print(rd)

        # datasets
        features2, labels2 = make_dataset2(1500,rd) 
        trainingfeatures2 = features2[300:]
        trainingtragets2 = labels2[300:]
        testingfeatures2 = features2[:300]
        testingtargets2 = labels2[:300]

        features1, labels1 = make_dataset1(1500,rd) 
        trainingfeatures1 = features1[300:]
        trainingtragets1 = labels1[300:]
        testingfeatures1 = features1[:300]
        testingtargets1 = labels1[:300]
        

        #QDA-DT1
        accuracy,std_deviation = test_method(trainingfeatures1,trainingtragets1,testingfeatures1,testingtargets1, lda=True)

        accuracy_dataset1_qda_tmp += accuracy
        std_deviation_dataset1_qda_tmp += std_deviation
        
        #_____________________________________________________________________________________________________________________


        # qda.fit(trainingfeatures1, trainingtragets1, lda=False)
        # proba_per_classes = qda.predict_proba(testingfeatures1)
        # prediction_classes = qda.predict(testingfeatures1)

        # accuracy_dataset1_qda_tmp += qda.compute_accuracy(prediction_classes,testingtargets1)

        # std_deviation_dataset1_qda_tmp += np.std(prediction_classes)
        # print(std_deviation_dataset1_qda_tmp)

        # name =  "---qda"+str(generation)
        # plot_boundary(name,qda,testingfeatures1, testingtargets1, title=name)



        # #QDA-DT2
        # qda.fit(trainingfeatures2, trainingtragets2, lda=False)
        # name =  "---qda"+str(generation)
        # plot_boundary(name,qda,testingfeatures2, testingtargets2, title=name)
        # proba_per_classes = qda.predict_proba(testingfeatures2)
        # prediction_classes = qda.predict(testingfeatures2)

        



        # #LDA-DT1
        # qda.fit(trainingfeatures1, trainingtragets1, lda=True)
        # proba_per_classes = qda.predict_proba(testingfeatures1)
        # prediction_classes = qda.predict(testingfeatures1)
        # name =  "---lda"+str(generation)
        # plot_boundary(name,qda,testingfeatures1, testingtargets1, title=name)

        # #LDA-DT2
        # qda.fit(trainingfeatures2, trainingtragets2, lda=True)
        # proba_per_classes = qda.predict_proba(testingfeatures2)
        # prediction_classes = qda.predict(testingfeatures2)
        # name =  "---lda"+str(generation)
        # plot_boundary(name,qda,testingfeatures2, testingtargets2, title=name)



    avg_accuracy_dataset1_qda = accuracy_dataset1_qda_tmp / 5 #repeat for others
    std_deviation_dataset1_qda = std_deviation_dataset1_qda_tmp / 5 #repeat for others
    print("acc : ",avg_accuracy_dataset1_qda ,"\nsdt : ", std_deviation_dataset1_qda)











    



    
