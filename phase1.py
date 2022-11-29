import os
import numpy as np
import pandas as pd
import sys
from tabulate import tabulate

import util
import datagen
import models

"""
EXPERIMENT 1
"""

def experiment1():
    print("\nRUNNING EXPERIMENT 1...")

    EXCLUDED_FEATURES = []
    LABEL = 14  # sensor 17
    WINDOW_SIZE = 50  # number of time frames for one prediction
    NUM_EPOCHS = 10  # specify number of epochs to train over
    BATCH_SIZE = 32  # specify batch size
    ERROR_MARGINS = [1, 3, 5]

    timer = util.TimeTracker()

    file = 'db1/S10_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS, ERROR_MARGINS)
    model.train()
    model.test()
    model.displayResults()

    timer.endTimer()
    print("Time elapsed: " + str(timer.elapsed) + " seconds")
    return model


"""
EXPERIMENT 2
"""

def experiment2():
    print("\nRUNNING EXPERIMENT 2...")

    EXCLUDED_FEATURES = []
    LABEL = 14  # sensor 17
    WINDOW_SIZE = 50  # number of time frames for one prediction
    NUM_EPOCHS = 10  # specify number of epochs to train over
    BATCH_SIZE = 32  # specify batch size
    ERROR_MARGINS = [1, 3, 5]

    timer = util.TimeTracker()

    file = 'db1/S7_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS, ERROR_MARGINS)
    model.train()
    model.test()
    model.displayResults()

    timer.endTimer()
    print("Time elapsed: " + str(timer.elapsed) + " seconds")
    return model


"""
EXPERIMENT 3
"""

def experiment3():
    print("\nRUNNING EXPERIMENT 3...")

    EXCLUDED_FEATURES = ['17']
    LABEL = 4  # sensor 7
    WINDOW_SIZE = 50  # number of time frames for one prediction
    NUM_EPOCHS = 10  # specify number of epochs to train over
    BATCH_SIZE = 32  # specify batch size
    ERROR_MARGINS = [1, 3, 5]

    timer = util.TimeTracker()

    file = 'db1/S10_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS, ERROR_MARGINS)
    model.train()
    model.test()
    model.displayResults()

    timer.endTimer()
    print("Time elapsed: " + str(timer.elapsed) + " seconds")
    return model


"""
EXPERIMENT 4
"""

def experiment4():
    print("\nRUNNING EXPERIMENT 4...")

    EXCLUDED_FEATURES = ['0', '1', '2', '3', '6', '8', '10', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    LABEL = 0  # sensor 5
    WINDOW_SIZE = 50  # number of time frames for one prediction
    NUM_EPOCHS = 10 # specify number of epochs to train over
    BATCH_SIZE = 32  # specify batch size
    ERROR_MARGINS = [1, 3, 5]

    timer = util.TimeTracker()

    file = 'db1/S7_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS, ERROR_MARGINS)
    model.train()
    model.test()
    model.displayResults()

    timer.endTimer()
    print("Time elapsed: " + str(timer.elapsed) + " seconds")
    return model


"""
EXPERIMENT 5
"""

def experiment5():
    print("\nRUNNING EXPERIMENT 5...")

    EXCLUDED_FEATURES = []
    LABEL = 14  # sensor 17
    WINDOW_SIZE = 50  # number of time frames for one prediction
    NUM_EPOCHS = 10  # specify number of epochs to train over
    BATCH_SIZE = 32  # specify batch size
    ERROR_MARGINS = [1, 3, 5]

    timer = util.TimeTracker()

    files = os.listdir("db1")
    files = ["db1/" + f for f in files]
    dataset = datagen.Parse(files, WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS, ERROR_MARGINS)
    model.train()
    model.test()
    model.displayResults()

    timer.endTimer()
    print("Time elapsed: " + str(timer.elapsed) + " seconds")
    return model


"""
EXPERIMENT 6
"""

def experiment6():
    EXCLUDED_FEATURES = []
    LABEL = 14  # sensor 17
    WINDOW_SIZE = 50  # number of time frames for one prediction
    NUM_EPOCHS = 10  # specify number of epochs to train over
    BATCH_SIZE = 32  # specify batch size
    ERROR_MARGINS = [1, 3, 5]
    file_norm_vals = {}

    def getTestingData(file):
        # generate time series data from csv data files
        print("Parsing data...")
        df = pd.read_csv(file).drop(columns=['Unnamed: 0', '5'])
        df = df.drop(columns=EXCLUDED_FEATURES)
        data = np.array(df, dtype=np.float32)
        scaled_data, data_min, data_max = util.scale_to_range(data, -1, 1)
        file_norm_vals[file] = (data_min, data_max)
        dataset = []
        for row in range(len(data) - (WINDOW_SIZE + 1)):
            features = np.delete(scaled_data[row:row + WINDOW_SIZE], [LABEL], 1)
            labels = [scaled_data[row + WINDOW_SIZE + 1][i] for i in [LABEL]]
            dataset.append((features, labels, file))
        print("Processed " + file)
        testing_features = []
        testing_labels = []
        testing_files = []
        for window in dataset: 
            testing_features.append(window[0])
            testing_labels.append(window[1])
            testing_files.append(window[2])
        return np.array(testing_features), np.array(testing_labels), np.array(testing_files)

    file = "phase1/experiment6.txt"
    sys.stdout = open(file, "w")

    print("\nRUNNING EXPERIMENT 6...")    

    timer = util.TimeTracker()

    file = 'db1/S10_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS, ERROR_MARGINS)
    model.train()

    files = os.listdir("db1")
    files = ["db1/" + f for f in files if f != 'S10_E3_A1_angles.csv']
    files = sorted(files)

    data = []
    headers = ["SUBJECT", "LOSS", "MEAN_ABS_ERROR"]
    headers.extend(["FRACTION OF PREDICTIONS W ERROR > " + str(e) for e in sorted(ERROR_MARGINS)])
    for file in files: 
        print("\nTESTING ON " + file + "...")
        dataset = getTestingData(file)
        results = model.getModel().evaluate(dataset[0], dataset[1], batch_size=BATCH_SIZE, verbose=0)
        predictions = model.getModel().predict(dataset[0], verbose=0)

        bounded_error_percentages = {}
        for margin in ERROR_MARGINS:
            bounded_error_percentages[margin] = 0
        
        unscaled_predictions = []
        unscaled_testing_labels = []
        for p in range(len(predictions)):
            norm_vals = file_norm_vals[file]

            unscaled_prediction = util.unscale_from_range(predictions[p], norm_vals[0], norm_vals[1], -1, 1)
            unscaled_predictions.append(unscaled_prediction)

            unscaled_label = util.unscale_from_range(dataset[1][p], norm_vals[0], norm_vals[1], -1, 1)
            unscaled_testing_labels.append(unscaled_label)

            for error_margin in bounded_error_percentages.keys():
                if abs(unscaled_label - unscaled_prediction) > error_margin:
                    bounded_error_percentages[error_margin] += 1

        for error_margin in bounded_error_percentages.keys():
            bounded_error_percentages[error_margin] /= len(predictions)

        subject = file.split("/")[1].split("_")[0]
        row = [subject, results[0], results[1]]
        for margin in sorted(ERROR_MARGINS):
            row.append(bounded_error_percentages[margin])
        data.append(row)
    
    print()
    print(tabulate(data, headers=headers, numalign="right"))

    timer.endTimer()
    print("\nTime elapsed: " + str(timer.elapsed) + " seconds")
    sys.stdout.close()
        
    return model


"""
RUN EXPERIMENTS
"""

# util.run_phase1_experiment(experiment1, "phase1/experiment1.txt", 10)
# util.run_phase1_experiment(experiment2, "phase1/experiment2.txt", 10)
# util.run_phase1_experiment(experiment3, "phase1/experiment3.txt", 10)
# util.run_phase1_experiment(experiment4, "phase1/experiment4.txt", 10)
# util.run_phase1_experiment(experiment5, "phase1/experiment5.txt", 1)
# experiment6()
 