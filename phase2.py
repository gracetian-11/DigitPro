import datagen
import models
import os
from tabulate import tabulate

"""
Experiment 1
"""

def experiment1(): 
    results = []
    for index_offset in range(0, 105, 5): 
        print("Training with Offset ", index_offset)
        EXCLUDED_FEATURES = []
        LABEL = 14
        WINDOW_SIZE = 50
        NUM_EPOCHS = 5
        BATCH_SIZE = 32
        OFFSET = index_offset

        ERROR_MARGIN = 5

        file = 'db1/S10_E3_A1_angles.csv'
        dataset = datagen.Parse([file], WINDOW_SIZE, EXCLUDED_FEATURES, [LABEL], OFFSET)

        model = models.MultiStepDense(dataset, BATCH_SIZE, NUM_EPOCHS)
        model.train()
        model.test()
        error_result = model.displayResults(verbose=False, error_margin=ERROR_MARGIN)
        results.append([OFFSET, error_result])
    
    print(tabulate(results, headers = ['Offset', '% Predictions w/ Error > ' + str(ERROR_MARGIN)]))
        

experiment1()