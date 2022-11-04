import datagen
import models
import os

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

    file = 'db1/S10_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
    model.train()
    model.test()
    model.displayResults()


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

    file = 'db1/S7_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
    model.train()
    model.test()
    model.displayResults()


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

    file = 'db1/S7_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
    model.train()
    model.test()
    model.displayResults()


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

    file = 'db1/S7_E3_A1_angles.csv'
    dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL])

    model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
    model.train()
    model.test()
    model.displayResults()


experiment1()
experiment2()
experiment3()
experiment4()