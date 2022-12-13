import math
import numpy as np
import random

import util

class Baselines:
    def __init__(self, dataset):
        self.dataset = dataset
        self.training_features = dataset.training_features
        self.training_labels = dataset.training_labels
        self.testing_features = dataset.testing_features

        self.random_selection, self.unscaled_random_selection = self.computeRandom()
        
    def computeRandom(self):
        # compute random selection
        print("Generating random selection baseline...")
        self.random_selection = []
        self.unscaled_random_selection = []
        for _ in self.testing_features:
            random_index = random.randrange(len(self.training_labels))
            norm_vals = self.dataset.file_norm_vals[self.dataset.training_files[random_index]]
            random_label = self.training_labels[random_index]
            unscaled_random_label = util.unscale_from_range(random_label, norm_vals[0], norm_vals[1], -1, 1)
            self.random_selection.append(random_label)
            self.unscaled_random_selection.append(unscaled_random_label)
        return self.random_selection, self.unscaled_random_selection
    
    def computeNearestNeighbor(self):
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
        return self.nearest_neighbor

    def getRandomSelection(self):
        return self.random_selection, self.unscaled_random_selection
    
    def getNearestNeighbor(self):
        return self.nearest_neighbor

    def __findDistance(self, features, target):
        if np.shape(features) != np.shape(target):
            return -1
        return math.sqrt(np.sum(np.absolute(features - target)))
