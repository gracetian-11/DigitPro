import pandas as pd
import numpy as np
import normalize
import math

class Parse:
    def __init__(self, files, window_size, excluded_features, label_indices):
        # dataset
        self.files = files
        self.file_norm_vals = {}
        self.dataset = []

        # training
        self.training_features = []
        self.training_labels = []
        self.training_files = []

        # testing
        self.testing_features = []
        self.testing_labels = []
        self.testing_files = []

        # generate time series data from csv data files
        print("Parsing data...")
        for file in self.files:
            df = pd.read_csv(file).drop(columns=['Unnamed: 0', '5'])
            df = df.drop(columns=excluded_features)
            data = np.array(df, dtype=np.float32)
            scaled_data, data_min, data_max = normalize.scale_to_range(data, -1, 1)
            self.file_norm_vals[file] = (data_min, data_max) 
            for row in range(len(data) - (window_size + 1)):
                features = np.delete(scaled_data[row:row + window_size], label_indices, 1)
                labels = [scaled_data[row + window_size + 1][i] for i in label_indices]
                self.dataset.append((features, labels, file))
            print("Processed " + file)

        # initialize training and testing datasets
        print("Generating training and testing datasets...")
        random_data = np.random.permutation(len(self.dataset))
        cutoff = math.floor(len(self.dataset) * 0.8)
        self.training_dataset, self.testing_dataset = random_data[:cutoff], random_data[cutoff:]
        for index in self.training_dataset: 
            window = self.dataset[index] 
            self.training_features.append(window[0])
            self.training_labels.append(window[1])
            self.training_files.append(window[2])
        self.training_features = np.array(self.training_features)
        self.training_labels = np.array(self.training_labels)
        for index in self.testing_dataset: 
            window = self.dataset[index] 
            self.testing_features.append(window[0])
            self.testing_labels.append(window[1])
            self.testing_files.append(window[2])
        self.testing_features = np.array(self.testing_features)
        self.testing_labels = np.array(self.testing_labels)
