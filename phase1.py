import os

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

    timer = util.TimeTracker()

    file = 'db1/S10_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
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

    timer = util.TimeTracker()

    file = 'db1/S7_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
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

    EXCLUDED_FEATURES = []
    LABEL = 4  # sensor 7
    WINDOW_SIZE = 50  # number of time frames for one prediction
    NUM_EPOCHS = 10  # specify number of epochs to train over
    BATCH_SIZE = 32  # specify batch size

    timer = util.TimeTracker()

    file = 'db1/S7_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
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

    timer = util.TimeTracker()

    file = 'db1/S7_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
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

    timer = util.TimeTracker()

    files = os.listdir("db1")
    files = ["db1/" + f for f in files]
    dataset = datagen.Parse(files, WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
    model.train()
    model.test()
    model.displayResults()

    timer.endTimer()
    print("Time elapsed: " + str(timer.elapsed) + " seconds")
    return model


"""
RUN EXPERIMENTS
"""

util.run_phase1_experiment(experiment1, "phase1/experiment1.txt", 10, 5)
util.run_phase1_experiment(experiment2, "phase1/experiment2.txt", 10, 5)
util.run_phase1_experiment(experiment3, "phase1/experiment3.txt", 10, 5)
util.run_phase1_experiment(experiment4, "phase1/experiment4.txt", 10, 5)
# util.run_phase1_experiment(experiment5, "phase1/experiment5.txt", 5, 5)