import numpy as np
import sys
from tabulate import tabulate
import util

import datagen
import models

"""
EXPERIMENT 1
"""

def experiment1():
    timer = util.TimeTracker()

    losses = []  # loss
    abs_errors = []  # mean absolute error
    error_percentages = []  # percentage of predictions > error_margin from ground truth
    results = []
    for index_offset in range(0, 105, 5): 
        print("\nOFFSET:", index_offset)
        LABEL = 14
        WINDOW_SIZE = 50
        NUM_EPOCHS = 5
        BATCH_SIZE = 32
        OFFSET = index_offset
        ERROR_MARGIN = 5

        file = 'db1/S10_E3_A1_angles.csv'
        dataset = datagen.Parse([file], WINDOW_SIZE, [], [LABEL], OFFSET)

        model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
        model.train()
        model.test()

        loss = model.results[0]
        mean_abs_error = model.results[1]
        error_percentage = model.displayResults(verbose=False, error_margin=ERROR_MARGIN)

        losses.append([OFFSET, loss])
        abs_errors.append([OFFSET, mean_abs_error])
        error_percentages.append([OFFSET, error_percentage])
        results.append([OFFSET, loss, mean_abs_error, error_percentage])
    
    print()
    headers = ["Offset", "Loss", "Mean Absolute Error", "% Predictions w/ Error > " + str(ERROR_MARGIN)]
    print(tabulate(results, headers = headers, numalign="right"))
    
    timer.endTimer()
    print("\nTotal time elapsed: " + str(timer.elapsed) + " seconds")

    return losses, abs_errors, error_percentages


"""
EXPERIMENT 2
"""

def experiment2():
    timer = util.TimeTracker()

    losses = []  # loss
    abs_errors = []  # mean absolute error
    error_percentages = []  # percentage of predictions > error_margin from ground truth
    results = []
    for window_size in range(10, 110, 10): 
        print("\nWINDOW SIZE:", window_size)
        LABEL = 14
        WINDOW_SIZE = window_size
        NUM_EPOCHS = 5
        BATCH_SIZE = 32
        OFFSET = 5
        ERROR_MARGIN = 5

        file = 'db1/S10_E3_A1_angles.csv'
        dataset = datagen.Parse([file], WINDOW_SIZE, [], [LABEL], OFFSET)

        model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
        model.train()
        model.test()

        loss = model.results[0]
        mean_abs_error = model.results[1]
        error_percentage = model.displayResults(verbose=False, error_margin=ERROR_MARGIN)

        losses.append([WINDOW_SIZE, loss])
        abs_errors.append([WINDOW_SIZE, mean_abs_error])
        error_percentages.append([WINDOW_SIZE, error_percentage])
        results.append([WINDOW_SIZE, loss, mean_abs_error, error_percentage])
    
    print()
    headers = ["Window Size", "Loss", "Mean Absolute Error", "% Predictions w/ Error > " + str(ERROR_MARGIN)]
    print(tabulate(results, headers = headers, numalign="right"))
    
    timer.endTimer()
    print("\nTotal time elapsed: " + str(timer.elapsed) + " seconds")

    return losses, abs_errors, error_percentages


"""
RUN EXPERIMENTS
"""

# util.run_phase2_experiment(experiment1, 10, "OFFSET")
util.run_phase2_experiment(experiment2, 10, "WINDOW SIZE")