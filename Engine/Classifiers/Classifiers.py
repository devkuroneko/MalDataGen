#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


try:
    import sys
    import logging

    from Engine.Classifiers.Algorithms.AdaBoost import AdaBoost

    from Engine.Classifiers.Algorithms.NaiveBayes import NaiveBayes
    from Engine.Classifiers.Algorithms.Perceptron import Perceptron

    from Engine.Classifiers.Algorithms.RandomForest import RandomForest
    from Engine.Classifiers.Algorithms.DecisionTree import DecisionTree

    from Engine.Classifiers.Algorithms.GaussianProcess import GaussianProcess

    from Engine.Classifiers.Algorithms.GradientBoosting import GradientBoosting
    from Engine.Classifiers.Algorithms.KMeansClustering import KMeansClustering

    from Engine.Classifiers.Algorithms.KNearestNeighbors import KNearestNeighbors

    from Engine.Classifiers.Algorithms.LinearRegression import LinearRegressionModel

    from Engine.Classifiers.Algorithms.SupportVectorMachine import SupportVectorMachine
    from Engine.Classifiers.Algorithms.SpectralClustering import SpectralClusteringModel

    from Engine.Classifiers.Algorithms.StochasticGradientDescent import StochasticGradientDescent
    from Engine.Classifiers.Algorithms.QuadraticDiscriminant import QuadranticDiscriminantAnalysis

except ImportError as error:
    print(error)
    sys.exit(-1)

def import_classifiers(function):

    def wrapper(self, *args, **kwargs):
        Classifiers.__init__(self, self.arguments)
        return function(self, *args, **kwargs)

    return wrapper


class Classifiers:
    dictionary_classifiers_name = ["RandomForest",
                                             "SupportVectorMachine",
                                             "KNN",
                                             #"GaussianPrecess",
                                             "DecisionTree",
                                             "DecisionTreeSubset",
                                             "RandomForestSubset",
                                             "ExtraTreesSubset",
                                             "RandomForestLight",
                                             #"AdaBoost", #TODO corrigir
                                             "NaiveBayes",
                                             #"QuadraticDiscriminant",
                                             #"Perceptron",
                                             "GradientBoosting",
                                             #"KMeansClustering",
                                             #"LinearRegression",
                                             "StochasticGradientDescent",
                                             "SGDClassifier",
                                             "PassiveAggressiveClassifier",
                                             "GaussianNB",
                                             "MLPClassifierSmall",
                                             #"SpectralClusteringModel"
                                             ]
    
    
    
    def __init__(self, arguments):

        dictionary_classifiers = {"RandomForest": RandomForest(arguments),
                                        "RandomForestSubset": RandomForest(arguments),
                                        "SupportVectorMachine": SupportVectorMachine(arguments),
                                        "KNN": KNearestNeighbors(arguments),
                                        "GaussianPrecess": GaussianProcess(arguments),
                                        "DecisionTree": DecisionTree(arguments),
                                        "DecisionTreeSubset": DecisionTree(arguments),
                                        # "AdaBoost": AdaBoost(arguments),
                                        "NaiveBayes": NaiveBayes(arguments),
                                        "QuadraticDiscriminant": QuadranticDiscriminantAnalysis(arguments),
                                        "Perceptron": Perceptron(arguments),
                                        "GradientBoosting": GradientBoosting(arguments),
                                        "KMeansClustering": KMeansClustering(arguments),
                                        "LinearRegression": LinearRegressionModel(arguments),
                                        "StochasticGradientDescent": StochasticGradientDescent(arguments),
                                        "SpectralClusteringModel": SpectralClusteringModel(arguments)}
        
        self._dictionary_classifiers_name = list()
        self._dictionary_classifiers = {}
        self._arguments = arguments

        for c in arguments.classifier: 
            if c in self.dictionary_classifiers_name and c in  dictionary_classifiers.keys():
                self._dictionary_classifiers_name.append(c)
                self._dictionary_classifiers[c] = dictionary_classifiers[c]
            elif getattr(arguments, "execution_mode", "normal") == "batches" and c in self.dictionary_classifiers_name:
                self._dictionary_classifiers_name.append(c)


    def get_trained_classifiers(self, x_samples_training, y_samples_training, dataset_type, input_dataset_shape):

        logging.info("")
        logging.info("Starting Training Classifiers")
        list_instance_classifiers = []

        for classifier_algorithm in self._dictionary_classifiers_name:

            classifier_model = self._dictionary_classifiers[classifier_algorithm]
            x_training, y_training = self._limit_subset_if_requested(
                classifier_algorithm,
                x_samples_training,
                y_samples_training,
            )
            list_instance_classifiers.append(classifier_model.get_model(x_training,
                                                                        y_training,
                                                                        dataset_type, input_dataset_shape))


        return list_instance_classifiers

    def _limit_subset_if_requested(self, classifier_algorithm, x_samples_training, y_samples_training):
        if classifier_algorithm not in {"DecisionTreeSubset", "RandomForestSubset"}:
            return x_samples_training, y_samples_training

        limit = int(getattr(self._arguments, "batch_classifier_subset_size", 100000))
        if limit <= 0:
            return x_samples_training, y_samples_training

        x_samples_training = x_samples_training[:limit]
        y_samples_training = y_samples_training[:limit]
        logging.info(
            "Training %s on subset: %d rows.",
            classifier_algorithm,
            len(y_samples_training),
        )
        return x_samples_training, y_samples_training
