from cProfile import label
import os
from pyexpat import model
from this import d
import pandas as pd
import numpy as np
import math
import tensorflow as tf

WINDOW_SIZE = 50  # number of time frames for one prediction
LABEL = 15  # sensor 17 corresponds to index 15 after dropping sensor 6 due to missing information

# generate time series from csv data files
directory = "db1"
dataset = [] # list of all (data, label) pairs
print("Generating dataset...")
for file in os.listdir(directory):
    subject = file.split('_')[0][1:]
    f = os.path.join(directory, file)
    df = pd.read_csv(f)
    df = df.drop(columns='5')
    data = np.array(df, dtype=np.float32)
    for row in range(len(data) - (WINDOW_SIZE + 1)):
        dataset.append((data[row:row + WINDOW_SIZE], data[row + WINDOW_SIZE + 1][LABEL]))  # (data, label) pair
    print("Processed " + f)
print("Generating dataset... done! :)")

# create training and testing datasets
print("Generating training and testing datasets...")
random_data = np.random.permutation(len(dataset))
cutoff = math.floor(len(dataset) * 0.8)
training_dataset, testing_dataset = random_data[:cutoff], random_data[cutoff:]
print("Generating training and testing datasets... done! :)")

# define model for experiment 1
model = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

# separate features and labels 
features = []
labels = []
for index in training_dataset: 
    window = dataset[index] 
    features.append(window[0])
    labels.append(window[1][15])
features = np.array(features)
labels = np.array(labels)

print("Compiling model...")
# compile the model with L2 loss and Adam optimizer
model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])
print("Compiling model... done! :)")

NUM_EPOCHS = 2 # specify number of epochs to train over
BATCH_SIZE = 32 # specify batch size 

print("Training Model...")
history = model.fit(features, labels, batch_size=BATCH_SIZE, epochs = NUM_EPOCHS)
print("Training Model... done! :)")

# separate features and labels 
features = []
labels = []
for index in testing_dataset: 
    window = dataset[index] 
    features.append(window[0])
    labels.append(window[1][15])
features = np.array(features)
labels = np.array(labels)

print("Testing model...")
results = model.evaluate(features, labels, batch_size=BATCH_SIZE)
print("Testing model... done! :)")
print("test loss, test acc:", results)

print("Generating predictions...")
predictions = model.predict(features[:10])
print("Generating predictions... done! :)")
print("Predictions: ")
print(predictions)
print("Ground Truth: ")
print(labels[:10])