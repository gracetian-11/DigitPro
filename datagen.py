import pandas as pd
import numpy as np
import normalize
import math

class Parse:
    def __init__(self, files, window_size, excluded_features, label_indices, offset = 1):
        # dataset
        self.files = files
        self.file_norm_vals = {}
        self.dataset = []

        # training
        self.training_features = []
        self.training_labels = []
        self.training_files = []

        # validation
        self.validation_features = []
        self.validation_labels = []
        self.validation_files = []

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
            for row in range(len(data) - (window_size + offset)):
                features = np.delete(scaled_data[row:row + window_size], label_indices, 1)
                labels = [scaled_data[row + window_size + offset][i] for i in label_indices]
                self.dataset.append((features, labels, file))
            print("Processed " + file)

        # split data into training, validation, and testing sets
        print("Generating training, validation, and testing datasets...")
        random_data = np.random.permutation(len(self.dataset))
        cutoff1 = math.floor(len(self.dataset) * 0.6)
        cutoff2 = math.floor(len(self.dataset) * 0.8)
        self.training_dataset, self.validation_dataset, self.testing_dataset = random_data[:cutoff1], random_data[cutoff1:cutoff2], random_data[cutoff2:]
        # create training set
        for index in self.training_dataset: 
            window = self.dataset[index] 
            self.training_features.append(window[0])
            self.training_labels.append(window[1])
            self.training_files.append(window[2])
        self.training_features = np.array(self.training_features)
        self.training_labels = np.array(self.training_labels)
        # create validation set
        for index in self.validation_dataset:
            window = self.dataset[index]
            self.validation_features.append(window[0])
            self.validation_labels.append(window[1])
            self.validation_files.append(window[2])
        self.validation_features = np.array(self.validation_features)
        self.validation_labels = np.array(self.validation_labels)
        # create testing set
        for index in self.testing_dataset: 
            window = self.dataset[index] 
            self.testing_features.append(window[0])
            self.testing_labels.append(window[1])
            self.testing_files.append(window[2])
        self.testing_features = np.array(self.testing_features)
        self.testing_labels = np.array(self.testing_labels)
