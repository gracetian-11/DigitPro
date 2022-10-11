import os
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