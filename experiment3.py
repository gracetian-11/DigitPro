import os
import pandas as pd
import numpy as np
import math
import tensorflow as tf
import normalize

LABEL = 0  # sensor 5 corresponds to index 0 after dropping other sensors and row numbers due to missing information
WINDOW_SIZE = 100  # number of time frames for one prediction
NUM_EPOCHS = 1 # specify number of epochs to train over
BATCH_SIZE = 32  # specify batch size 

# generate time series from csv data files
directory = "db1"
dataset = [] # list of all (data, label) pairs
scaled_dataset = [] # list of all scaled (data, label) pairs
file_count = 0 # counter for files to be processed
subject_norm_vals = {} # dicitonary to store max/min values for each subject data for unscaling data 
print("Generating dataset...")
for file in os.listdir(directory):
    if file != "S7_E3_A1_angles.csv": continue
    subject = file.split('_')[0][1:]
    f = os.path.join(directory, file)
    df = pd.read_csv(f)
    df = df.drop(columns='5')
    df = df.drop(columns=['Unnamed: 0', '0', '1', '2', '3', '6', '8', '10', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'])
    
    data = np.array(df, dtype=np.float32)
    scaled_data, data_min, data_max = normalize.scale_to_range(data, -1, 1)
    subject_norm_vals[subject] = (data_min, data_max) 
    for row in range(len(data) - (WINDOW_SIZE + 1)):
        if data[row+WINDOW_SIZE +1][LABEL] > 360 or data[row+WINDOW_SIZE +1][LABEL] < -360:
            print(data[row+WINDOW_SIZE +1][LABEL])
        dataset.append((np.delete(data[row:row + WINDOW_SIZE], LABEL, 1), data[row + WINDOW_SIZE + 1][LABEL], subject))  # (data, label, subject) pair
        scaled_dataset.append((np.delete(scaled_data[row:row + WINDOW_SIZE], LABEL, 1), scaled_data[row + WINDOW_SIZE + 1][LABEL], subject))  # scaled (data, label, subject) pair
    print("Processed " + f)
    break
print("Generating dataset... done! :)")

# create training and testing datasets from scaled_dataset
print("Generating training and testing datasets...")
random_data = np.random.permutation(len(scaled_dataset))
cutoff = math.floor(len(scaled_dataset) * 0.8)
training_dataset, testing_dataset = random_data[:cutoff], random_data[cutoff:]
print("Generating training and testing datasets... done! :)")

# define model for experiment 1
model = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=250, activation='tanh'),
    tf.keras.layers.Dense(units=250, activation='tanh'),
    tf.keras.layers.Dense(units=1, activation='tanh'),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

# separate features and labels for training and testing datasets
training_features, training_labels, training_subjects, testing_features, testing_labels, testing_subjects = [], [], [], [], [], []
for index in training_dataset: 
    window = scaled_dataset[index] 
    training_features.append(window[0])
    training_labels.append(window[1])
    training_subjects.append(window[2])
training_features = np.array(training_features)
training_labels = np.array(training_labels)
for index in testing_dataset: 
    window = scaled_dataset[index] 
    testing_features.append(window[0])
    testing_labels.append(window[1])
    testing_subjects.append(window[2])
testing_features = np.array(testing_features)
testing_labels = np.array(testing_labels)
print("Len testing_features: ", testing_features.size)

# compile the model with L1 loss and Adam optimizer
print("Compiling model...")
model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])
print("Compiling model... done! :)")

print("Training model...")
history = model.fit(training_features, training_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
print("Training model... done! :)")

print("Testing model...")
results = model.evaluate(testing_features, testing_labels, batch_size=BATCH_SIZE, verbose=0)
print("Testing model... done! :)")
print("test loss, test acc:", results)

print("Generating predictions...")
predictions = model.predict(testing_features[:20], verbose=0)
unscaled_predictions = []
unscaled_testing_labels = [] 
for p in range(len(predictions)):
    norm_vals = subject_norm_vals[testing_subjects[p]]
    unscaled_predictions.append(normalize.unscale_from_range(predictions[p], norm_vals[0], norm_vals[1], -1, 1))
    unscaled_testing_labels.append(normalize.unscale_from_range(testing_labels[p], norm_vals[0], norm_vals[1], -1, 1))
print("Predictions: ", [p[0][0] for p in unscaled_predictions])
print("Ground Truth", unscaled_testing_labels)
print("Generating predictions... done! :)")