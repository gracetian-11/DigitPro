import numpy as np
import sys
from tabulate import tabulate
import time

class TimeTracker:
    def __init__(self):
        self.start = time.time()
        self.end = 0
        self.elapsed = 0

    def endTimer(self):
        self.end = time.time()
        self.elapsed = self.end - self.start

def scale_to_range(data, left_bound, right_bound):
    return ((data - np.min(data)) / (np.max(data) - np.min(data))) * (right_bound - left_bound) + left_bound, np.min(data), np.max(data)

def unscale_from_range(data, min_val, max_val, left_bound, right_bound): 
    return (data - left_bound) / (right_bound - left_bound) * (max_val - min_val) + min_val

def run_phase1_experiment(function, file, iterations, error_margin):
    experiment = str(function).split(" ")[1]

    sys.stdout = open(file, "w")
    print(experiment.upper())
    data = []
    losses = []  # loss
    abs_errors = []  # mean absolute error
    error_percentages = []  # percentage of predictions > error_margin from ground truth
    for iter in range(iterations):
        model = function()
        results = []

        loss = model.results[0]
        mean_abs_error = model.results[1]
        error_percentage = model.bounded_error_percentage

        losses.append(loss)
        abs_errors.append(mean_abs_error)
        error_percentages.append(error_percentage)

        results = [iter, loss, mean_abs_error, error_percentage]
        data.append(results)
    data.append(["MEAN", np.mean(losses), np.mean(abs_errors), np.mean(error_percentages)])
    data.append(["STD", np.std(losses), np.std(abs_errors), np.std(error_percentages)])
    sys.stdout.close()

    headers = ["ITERATION", "LOSS", "MEAN_ABS_ERROR", "% PREDICTIONS > " + str(error_margin)]
    summary_file = file.split(".")[0] + "_summary.txt"
    file = open(summary_file, "w")
    file.write(tabulate(data, headers=headers, numalign="right"))

def run_phase2_experiment(function, iterations, variable):
    experiment = str(function).split(" ")[1]
    dir = "phase2/" + experiment + "/"

    aggregate_losses = {}
    aggregate_abs_errors = {}
    aggregate_error_percentages = {}
    for iteration in range(iterations):
        file = dir + "iteration" + str(iteration) + ".txt"
        sys.stdout = open(file, "w")
        print(experiment.upper())
        losses, abs_errors, error_percentages = function()
        sys.stdout.close()

        for var, metric in losses:
            if var not in aggregate_losses.keys():
                aggregate_losses[var] = []
            aggregate_losses[var].append(metric)

        for var, metric in abs_errors:
            if var not in aggregate_abs_errors.keys():
                aggregate_abs_errors[var] = []
            aggregate_abs_errors[var].append(metric)

        for var, metric in error_percentages:
            if var not in aggregate_error_percentages.keys():
                aggregate_error_percentages[var] = []
            aggregate_error_percentages[var].append(metric)

    
    headers = [str(i) for i in range(iterations)]
    headers.insert(0, variable)
    headers.extend(["MEAN", "STD"])

    losses = []
    for var in aggregate_losses:
        row = [var]
        row.extend(aggregate_losses[var])
        row.append(np.mean(aggregate_losses[var]))
        row.append(np.std(aggregate_losses[var]))
        losses.append(row)
    
    file = dir + "summary_losses.txt"
    f = open(file, "w")
    f.write(tabulate(losses, headers=headers, numalign="right"))

    abs_errors = []
    for var in aggregate_abs_errors:
        row = [var]
        row.extend(aggregate_abs_errors[var])
        row.append(np.mean(aggregate_abs_errors[var]))
        row.append(np.std(aggregate_abs_errors[var]))
        abs_errors.append(row)
    
    file = dir + "summary_abs_errors.txt"
    f = open(file, "w")
    f.write(tabulate(abs_errors, headers=headers, numalign="right"))

    error_percentages = []
    for var in aggregate_error_percentages:
        row = [var]
        row.extend(aggregate_error_percentages[var])
        row.append(np.mean(aggregate_error_percentages[var]))
        row.append(np.std(aggregate_error_percentages[var]))
        error_percentages.append(row)
    
    file = dir + "summary_error_percentages.txt"
    f = open(file, "w")
    f.write(tabulate(error_percentages, headers=headers, numalign="right"))