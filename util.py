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


"""
compares results from model and baseline predictions to ground truth, 
writing results to the specified file

ground_truth (list): correct labels
results (dict): dictionary mapping to type of result (e.g. model, baseline) to its list of predictions
error margins (list): list of integer values representing acceptable errors to track
"""
def compare_baselines(ground_truth, unscaled_ground_truth, results, unscaled_results, error_margins):
    headers = ["", "MEAN ERROR FROM GROUND TRUTH", "STD OF ERRORS FROM GROUND TRUTH"]
    headers.extend(["FRACTION OF PREDICTIONS W ERROR > " + str(e) for e in sorted(error_margins)])

    data = []
    for type in results.keys():
        if len(ground_truth) != len(results[type]):
            return

        result = results[type]

        bounded_error_percentages = {}
        for e in error_margins:
            bounded_error_percentages[e] = 0

        errors = []
        for i in range(len(ground_truth)):
            error = abs(ground_truth[i] - result[i])
            errors.append(error)
            unscaled_error = abs(unscaled_ground_truth[i] - unscaled_results[type][i])
            for e in error_margins:
                if unscaled_error > e:
                    bounded_error_percentages[e] += 1

        errors_mean = np.mean(errors)
        errors_std = np.std(errors)
        for e in bounded_error_percentages.keys():
            bounded_error_percentages[e] /= len(ground_truth)

        comparisons = [type, errors_mean, errors_std]
        comparisons.extend([bounded_error_percentages[e] for e in sorted(error_margins)])
        data.append(comparisons)

    print(tabulate(data, headers=headers, numalign="right"))

    return headers, data


def run_phase1_experiment(function, file, iterations):
    experiment = str(function).split(" ")[1]

    sys.stdout = open(file, "w")
    print(experiment.upper())
    data = []
    losses = []  # loss
    abs_errors = []  # mean absolute error
    error_percentages = []  # percentage of predictions > error_margin from ground truth
    error_margins = []
    for iter in range(iterations):
        model = function()
        results = []
        error_margins = sorted(model.bounded_error_percentages.keys())

        loss = model.results[0]
        mean_abs_error = model.results[1]
        percentages = [model.bounded_error_percentages[margin] for margin in error_margins]

        losses.append(loss)
        abs_errors.append(mean_abs_error)
        error_percentages.append(percentages)

        results = [iter, loss, mean_abs_error]
        results.extend(percentages)
        data.append(results)
    sys.stdout.close()

    means = ["MEAN", np.mean(losses), np.mean(abs_errors)]
    stds = ["STD", np.std(losses), np.std(abs_errors)]
    for i in range(len(error_margins)):
        means.append(np.mean([e[i] for e in error_percentages]))
        stds.append(np.std([e[i] for e in error_percentages]))
    data.append(means)
    data.append(stds)

    headers = ["ITERATION", "LOSS", "MEAN_ABS_ERROR"]
    headers.extend(["FRACTION OF PREDICTIONS W/ ERROR > " + str(error_margin) for error_margin in error_margins])
    summary_file = file.split(".")[0] + "_summary.txt"
    file = open(summary_file, "w")
    file.write(tabulate(data, headers=headers, numalign="right"))
    file.close()


def run_phase2_experiment(function, iterations, variable):
    experiment = str(function).split(" ")[1]
    dir = "phase2/" + experiment + "/"

    aggregate_losses = {}  # key: variable (offset/window size), value: list of results over iterations
    aggregate_abs_errors = {}  # key: variable (offset/window size), value: list of results over iterations
    aggregate_error_percentages = {}  # key: error margins, value: dictionary mapping variable (offset/window size) to list of results over iterations
    for iteration in range(iterations):
        file = dir + "iteration" + str(iteration) + ".txt"
        sys.stdout = open(file, "w")
        print(experiment.upper())
        losses, abs_errors, error_percentages, error_margins = function()
        sys.stdout.close()

        for var, metric in losses:
            if var not in aggregate_losses.keys():
                aggregate_losses[var] = []
            aggregate_losses[var].append(metric)

        for var, metric in abs_errors:
            if var not in aggregate_abs_errors.keys():
                aggregate_abs_errors[var] = []
            aggregate_abs_errors[var].append(metric)

        for error_margin in error_margins:
            if error_margin not in aggregate_error_percentages:
                aggregate_error_percentages[error_margin] = {}

        for ep in error_percentages:
            var = ep[0]
            metrics = ep[1:]
            for i in range(len(metrics)):
                if var not in aggregate_error_percentages[error_margins[i]]:
                    aggregate_error_percentages[error_margins[i]][var] = []
                aggregate_error_percentages[error_margins[i]][var].append(metrics[i])
    
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

    file = dir + "summary_error_percentages.txt"
    f = open(file, "w")
    for error_margin in sorted(aggregate_error_percentages.keys()):
        f.write("FRACTION OF PREDICTIONS W/ ERROR > " + str(error_margin) + ":\n")
        error_percentages = []
        for var in aggregate_error_percentages[error_margin]:
            row = [var]
            row.extend(aggregate_error_percentages[error_margin][var])
            row.append(np.mean(aggregate_error_percentages[error_margin][var]))
            row.append(np.std(aggregate_error_percentages[error_margin][var]))
            error_percentages.append(row)
        f.write(tabulate(error_percentages, headers=headers, numalign="right"))
        f.write("\n\n")
    f.close()
