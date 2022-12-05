import math
import numpy as np
import random

class Baselines:
    def __init__(self, dataset):
        self.training_features = dataset.training_features
        self.training_labels = dataset.training_labels
        self.testing_features = dataset.testing_features

        # compute random selection
        print("Generating random selection baseline...")
        self.random_selection = []
        for _ in self.testing_features:
            random_label = random.choice(self.training_labels)
            self.random_selection.append(random_label)

        # compute nearest neighbor
        print("Generating nearest neighbor baseline...")
        self.nearest_neighbor = []
        for test_feature in self.testing_features:
            min_distance = float('inf')
            min_label = 0
            for train_index in range(len(self.training_features)):
                train_feature = self.training_features[train_index]
                dist = self.__findDistance(test_feature, train_feature)
                if dist == -1: continue
                if dist < min_distance:
                    min_distance = dist
                    min_label = self.training_labels[train_index]
            self.nearest_neighbor.append(min_label)

    def getRandomSelection(self):
        return self.random_selection
    
    def getNearestNeighbor(self):
        return self.nearest_neighbor

    def __findDistance(self, features, target):
        if np.shape(features) != np.shape(target):
            return -1
        return math.sqrt(np.sum(np.absolute(features - target)))
