from tabulate import tabulate

import datagen
import models
import util

"""
EXPERIMENT 1
"""

def experiment1():
    timer = util.TimeTracker()

    LABEL = 14
    WINDOW_SIZE = 50
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    ERROR_MARGINS = [1, 3, 5]

    losses = []  # loss
    abs_errors = []  # mean absolute error
    error_percentages = []  # percentage of predictions > error_margin from ground truth
    results = []

    for index_offset in range(0, 105, 5): 
        print("\nOFFSET:", index_offset)
        
        OFFSET = index_offset

        file = 'db1/S10_E3_A1_angles.csv'
        dataset = datagen.Parse([file], WINDOW_SIZE, [], [LABEL], OFFSET)

        model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS, ERROR_MARGINS)
        model.train()
        model.test()

        loss = model.results[0]
        mean_abs_error = model.results[1]
        bounded_error_percentages = model.bounded_error_percentages
        percentages = [bounded_error_percentages[margin] for margin in sorted(bounded_error_percentages.keys())]

        losses.append([OFFSET, loss])
        abs_errors.append([OFFSET, mean_abs_error])
        error_percentage = [OFFSET]
        error_percentage.extend(percentages)
        error_percentages.append(error_percentage)

        result = [OFFSET, loss, mean_abs_error]
        result.extend(percentages)
        results.append(result)
    
    print()
    headers = ["Offset", "Loss", "Mean Absolute Error"]
    headers.extend(["Fraction of Predictions w/ Error > " + str(e) for e in ERROR_MARGINS])
    print(tabulate(results, headers = headers, numalign="right"))
    
    timer.endTimer()
    print("\nTotal time elapsed: " + str(timer.elapsed) + " seconds")

    return losses, abs_errors, error_percentages, ERROR_MARGINS


"""
EXPERIMENT 2
"""

def experiment2():
    timer = util.TimeTracker()

    LABEL = 14
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    OFFSET = 5
    ERROR_MARGINS = [1, 3, 5]

    losses = []  # loss
    abs_errors = []  # mean absolute error
    error_percentages = []  # percentage of predictions > error_margin from ground truth
    results = []
    window_sizes = [1]
    window_sizes.extend([i for i in range(10, 110, 10)])
    for window_size in window_sizes: 
        print("\nWINDOW SIZE:", window_size)
        
        WINDOW_SIZE = window_size

        file = 'db1/S10_E3_A1_angles.csv'
        dataset = datagen.Parse([file], WINDOW_SIZE, [], [LABEL], OFFSET)

        model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS, ERROR_MARGINS)
        model.train()
        model.test()

        loss = model.results[0]
        mean_abs_error = model.results[1]
        bounded_error_percentages = model.bounded_error_percentages
        percentages = [bounded_error_percentages[margin] for margin in sorted(bounded_error_percentages.keys())]

        losses.append([WINDOW_SIZE, loss])
        abs_errors.append([WINDOW_SIZE, mean_abs_error])
        error_percentage = [WINDOW_SIZE]
        error_percentage.extend(percentages)
        error_percentages.append(error_percentage)

        result = [WINDOW_SIZE, loss, mean_abs_error]
        result.extend(percentages)
        results.append(result)

    print()
    headers = ["Window Size", "Loss", "Mean Absolute Error"]
    headers.extend(["Fraction of Predictions w/ Error > " + str(e) for e in ERROR_MARGINS])
    print(tabulate(results, headers = headers, numalign="right"))
    
    timer.endTimer()
    print("\nTotal time elapsed: " + str(timer.elapsed) + " seconds")

    return losses, abs_errors, error_percentages, ERROR_MARGINS


"""
RUN EXPERIMENTS
"""

# util.run_phase2_experiment(experiment1, 10, "OFFSET")
# util.run_phase2_experiment(experiment2, 10, "WINDOW SIZE")