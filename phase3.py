import sys

import baselines
import datagen
import models
import util

"""
EXPERIMENT 1
"""

def experiment1():
    file = "phase3/experiment1.txt"
    sys.stdout = open(file, "w") 
    print("\nRUNNING EXPERIMENT 1...")

    EXCLUDED_FEATURES = []
    LABEL = 14  # sensor 17
    WINDOW_SIZE = 50  # number of time frames for one prediction
    NUM_EPOCHS = 10  # specify number of epochs to train over
    BATCH_SIZE = 32  # specify batch size 
    NUM_MIXES = 1

    timer = util.TimeTracker()

    file = 'db1/S10_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MixtureDensity(dataset, BATCH_SIZE, NUM_EPOCHS, NUM_MIXES)
    model.train()
    model.test()
    model.displayResults()

    timer.endTimer()
    print("Time elapsed: " + str(timer.elapsed) + " seconds")
    sys.stdout.close()
    return model


"""
EXPERIMENT 2
"""

def experiment2():
    file = "phase3/experiment2.txt"
    sys.stdout = open(file, "w") 
    print("\nRUNNING EXPERIMENT 2...")

    EXCLUDED_FEATURES = []
    LABEL = 14  # sensor 17
    WINDOW_SIZE = 50  # number of time frames for one prediction
    NUM_EPOCHS = 10  # specify number of epochs to train over
    BATCH_SIZE = 32  # specify batch size 
    NUM_MIXES = 5

    timer = util.TimeTracker()

    file = 'db1/S10_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MixtureDensity(dataset, BATCH_SIZE, NUM_EPOCHS, NUM_MIXES)
    model.train()
    model.test()
    model.displayResults()

    timer.endTimer()
    print("Time elapsed: " + str(timer.elapsed) + " seconds")
    sys.stdout.close()
    return model


"""
EXPERIMENT 3
"""

def experiment3():
    file = "phase3/experiment3.txt"
    sys.stdout = open(file, "w")
    print("\nRUNNING EXPERIMENT 3...")
    
    EXCLUDED_FEATURES = []
    LABEL = 14  # sensor 17
    WINDOW_SIZE = 50  # number of time frames for one prediction
    NUM_EPOCHS = 10  # specify number of epochs to train over
    BATCH_SIZE = 32  # specify batch size 
    NUM_MIXES = 5
    ERROR_MARGINS = [1, 3, 5]

    timer = util.TimeTracker()

    file = 'db1/S10_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    base = baselines.Baselines(dataset)
    rand_baseline, unscaled_rand_baseline = base.getRandomSelection()

    model = models.MixtureDensity(dataset, BATCH_SIZE, NUM_EPOCHS, NUM_MIXES)
    model.train()
    model.test()

    results = {"MODEL": model.prediction_values, "RANDOM": rand_baseline}
    unscaled_results = {"MODEL": model.unscaled_predictions, "RANDOM": unscaled_rand_baseline}
    util.compare_baselines(ground_truth=dataset.testing_labels, 
                           unscaled_ground_truth=model.unscaled_testing_labels,
                           results=results, 
                           unscaled_results=unscaled_results,
                           error_margins=ERROR_MARGINS)

    timer.endTimer()
    print("Time elapsed: " + str(timer.elapsed) + " seconds")
    sys.stdout.close()
    return model


"""
RUN EXPERIMENTS
"""

# experiment1()
# experiment2()
# experiment3()
