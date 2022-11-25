import os
import util
import datagen
import models
import tensorflow as tf
import tensorflow_probability as tfd

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
    return model

# experiment1()
experiment2()
