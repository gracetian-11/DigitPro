import tensorflow as tf
from tabulate import tabulate
import keras_tuner as kt

import util

class MultiStepDense:
    def __init__(self, dataset, batch_size, num_epochs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
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

    def train(self):
        print("Training model...")
        self.model.fit(self.dataset.training_features, self.dataset.training_labels, batch_size=self.batch_size, epochs=self.num_epochs, verbose=0)

    def test(self):
        print("Testing model...")
        self.results = self.model.evaluate(self.dataset.testing_features, self.dataset.testing_labels, batch_size=self.batch_size, verbose=0)

        print("Generating predictions...")
        self.predictions = self.model.predict(self.dataset.testing_features, verbose=0)

    def displayResults(self, verbose=True, num_to_display=20, error_margin=5):
        for i in range(len(self.model.metrics_names)):
            print(self.model.metrics_names[i] + ": " + str(self.results[i]))
        unscaled_predictions = []
        unscaled_testing_labels = []
        count = 0
        for p in range(len(self.predictions)):
            norm_vals = self.dataset.file_norm_vals[self.dataset.testing_files[p]]
            unscaled_prediction = util.unscale_from_range(self.predictions[p], norm_vals[0], norm_vals[1], -1, 1)
            unscaled_predictions.append(unscaled_prediction)
            unscaled_label = util.unscale_from_range(self.dataset.testing_labels[p], norm_vals[0], norm_vals[1], -1, 1)
            unscaled_testing_labels.append(unscaled_label)
            if abs(unscaled_label - unscaled_prediction) > error_margin:
                count += 1
        display_data = []
        for i in range(num_to_display):
            p = unscaled_predictions[i][0][0]
            l = unscaled_testing_labels[i][0]
            display_data.append([p, l, abs(p - l)])
        if verbose:
            print(tabulate(display_data, headers=['Predictions', 'Ground Truth', 'Error']))
        print("% predictions with error > " + str(error_margin) + ": " + str(count / len(self.predictions)))
        self.bounded_error_percentage = count / len(self.predictions)
        return self.bounded_error_percentage
        