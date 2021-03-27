import pickle
#from sklearn.tree import DecisionTreeClassifier
import numpy as np
import math

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n_train = len(y)
        self.classifier_list = []
        self.alpha = []
        weight = np.ones((n_train, ))/n_train
        for i in range(self.n_weakers_limit):
            #create a weak classifier
            self.classifier_list.append(self.weak_classifier(max_depth=1))
            self.classifier_list[i].fit(X, y, sample_weight=weight)
            pred_ans = self.classifier_list[i].predict(X)
            #calc the miss/error rate
            miss = [int(x) for x in pred_ans != y]
            error = np.dot(miss, weight)
            #calc the weak classifier weight
            alp_i = 0.5 * np.log((1-error)/np.maximum(error, 10 ** -10))
            self.alpha.append(alp_i)
            z_t = 0
            for j in range(n_train):
                weight[j] = weight[j] * math.exp(-alp_i*y[j]*pred_ans[j])
                z_t += weight[j]
            weight = weight / np.maximum(z_t, 10 ** -10)

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        n_test = len(X)
        sum_y = np.zeros((n_test, ))
        for i in range(self.n_weakers_limit):
            y_i = self.alpha[i] * self.classifier_list[i].predict(X)
            sum_y += y_i
        return sum_y

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        sum_y = self.predict_scores(X)
        for i in range(len(sum_y)):
            if sum_y[i] > 0:
                sum_y[i] = 1
            else:
                sum_y[i] = -1;
        return sum_y

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
