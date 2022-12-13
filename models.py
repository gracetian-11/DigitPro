import tensorflow as tf
from tabulate import tabulate
import mdn
import numpy as np

import util

class MultiStepDense:
    def __init__(self, dataset, batch_size, num_epochs, error_margins=[1, 3, 5]):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.bounded_error_percentages = {}
        for margin in error_margins:
            self.bounded_error_percentages[margin] = 0

        self.model = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='tanh'),
            tf.keras.layers.Dense(units=128, activation='tanh'),
            tf.keras.layers.Dense(units=1, activation='tanh'),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ])
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                            optimizer=tf.keras.optimizers.Adam(),
                            metrics=[tf.keras.metrics.MeanAbsoluteError()])

    def getModel(self):
        return self.model

    def train(self):
        print("Training model...")
        self.model.fit(self.dataset.training_features, self.dataset.training_labels, batch_size=self.batch_size, epochs=self.num_epochs, verbose=0)

    def test(self):
        print("Testing model...")
        self.results = self.model.evaluate(self.dataset.testing_features, self.dataset.testing_labels, batch_size=self.batch_size, verbose=0)

        print("Generating predictions...")
        self.predictions = self.model.predict(self.dataset.testing_features, verbose=0)

        self.unscaled_predictions = []
        self.unscaled_testing_labels = []
        for p in range(len(self.predictions)):
            norm_vals = self.dataset.file_norm_vals[self.dataset.testing_files[p]]

            unscaled_prediction = util.unscale_from_range(self.predictions[p], norm_vals[0], norm_vals[1], -1, 1)
            self.unscaled_predictions.append(unscaled_prediction)

            unscaled_label = util.unscale_from_range(self.dataset.testing_labels[p], norm_vals[0], norm_vals[1], -1, 1)
            self.unscaled_testing_labels.append(unscaled_label)

            for error_margin in self.bounded_error_percentages.keys():
                if abs(unscaled_label - unscaled_prediction) > error_margin:
                    self.bounded_error_percentages[error_margin] += 1

        for error_margin in self.bounded_error_percentages.keys():
            self.bounded_error_percentages[error_margin] /= len(self.predictions)

    def displayResults(self, verbose=True, num_to_display=20):
        # display results
        for i in range(len(self.model.metrics_names)):
            print(self.model.metrics_names[i] + ": " + str(self.results[i]))

        # display prediction, ground truth, and error for num_to_display predictions
        if verbose:
            display_data = []
            for i in range(num_to_display):
                p = self.unscaled_predictions[i][0][0]
                l = self.unscaled_testing_labels[i][0]
                display_data.append([p, l, abs(p - l)])
            print(tabulate(display_data, headers=['Predictions', 'Ground Truth', 'Error']))
        
        # display the percent of predictions in error_margins
        for error_margin in sorted(self.bounded_error_percentages.keys()):
            print("fraction of predictions with error > " + str(error_margin) + ": " + str(self.bounded_error_percentages[error_margin]))

        return self.bounded_error_percentages
        

class MixtureDensity:
    def __init__(self, dataset, batch_size, num_epochs, num_mixes, error_margins=[1, 3, 5]):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_mixes = num_mixes
        self.bounded_error_percentages = {}
        for margin in error_margins:
            self.bounded_error_percentages[margin] = 0
        self.mean_squared_error = 0
        self.mean_absolute_error = 0

        self.model = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='tanh'),
            tf.keras.layers.Dense(units=128, activation='tanh'),
            mdn.MDN(1, self.num_mixes)
        ])

        self.model.compile(loss=mdn.get_mixture_loss_func(1, self.num_mixes),
                            optimizer=tf.keras.optimizers.Adam())

    def train(self):
        print("Training model...")
        self.model.fit(self.dataset.training_features, self.dataset.training_labels, batch_size=self.batch_size, epochs=self.num_epochs, verbose=0)

    def test(self):
        print("Testing model...")
        self.results = self.model.evaluate(self.dataset.testing_features, self.dataset.testing_labels, batch_size=self.batch_size, verbose=0)

        print("Generating predictions...")
        self.predictions = self.model.predict(self.dataset.testing_features, verbose=0)

        self.mus = np.apply_along_axis((lambda a: a[:self.num_mixes]), 1, self.predictions)
        self.sigs = np.apply_along_axis((lambda a: a[self.num_mixes:2*self.num_mixes]), 1, self.predictions)
        self.pis = np.apply_along_axis((lambda a: mdn.softmax(a[2*self.num_mixes:])), 1, self.predictions)

        self.prediction_values = []
        self.unscaled_predictions = []
        self.unscaled_testing_labels = []
        for p in range(len(self.predictions)):
            norm_vals = self.dataset.file_norm_vals[self.dataset.testing_files[p]]

            curr_prediction = 0 
            for mu in range(len(self.mus[p])):
                curr_prediction += self.mus[p][mu] * self.pis[p][mu]
            self.prediction_values.append(curr_prediction)

            self.mean_squared_error += abs(curr_prediction - self.dataset.testing_labels[p][0]) ** 2
            self.mean_absolute_error += abs(curr_prediction - self.dataset.testing_labels[p][0])

            unscaled_prediction = util.unscale_from_range(curr_prediction, norm_vals[0], norm_vals[1], -1, 1)
            self.unscaled_predictions.append(unscaled_prediction)

            unscaled_label = util.unscale_from_range(self.dataset.testing_labels[p], norm_vals[0], norm_vals[1], -1, 1)
            self.unscaled_testing_labels.append(unscaled_label)
            
            for error_margin in self.bounded_error_percentages.keys():
                if abs(unscaled_label - unscaled_prediction) > error_margin:
                    self.bounded_error_percentages[error_margin] += 1

        for error_margin in self.bounded_error_percentages.keys():
            self.bounded_error_percentages[error_margin] /= len(self.predictions)

        self.mean_squared_error /= len(self.predictions)
        self.mean_absolute_error /= len(self.predictions)

    def displayResults(self, verbose=True, num_to_display=20):
        print("mean_squared_error: " + str(self.mean_squared_error))
        print("mean_absolute_error: " + str(self.mean_absolute_error))

        # display prediction, ground truth, and error for num_to_display predictions
        if verbose:
            display_data = []
            for i in range(num_to_display):
                p = self.unscaled_predictions[i]
                l = self.unscaled_testing_labels[i][0]
                display_data.append([p, l, abs(p - l), self.mus[i], self.sigs[i], self.pis[i]])
            print(tabulate(display_data, headers=['Prediction', 'Ground Truth', 'Error', 'Mu', 'Sigma', 'Pi']))
        
        # display the percent of predictions in error_margins
        for error_margin in sorted(self.bounded_error_percentages.keys()):
            print("fraction of predictions with error > " + str(error_margin) + ": " + str(self.bounded_error_percentages[error_margin]))

        return self.bounded_error_percentages
        