"""
Module for data visualization in db1.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

directory = "db1"
info = open('dataviz.txt', 'w')
info.write('')
for file in os.listdir(directory):
    subject = file.split('_')[0][1:]
    f = os.path.join(directory, file)
    df = pd.read_csv(f)
    df = df.drop(columns=['Unnamed: 0'])
    df.hist(bins=36)
    path = 'dataviz/S' + subject
    plt.savefig(path)

    std = pd.Series(df.std(), name='STD')
    range = pd.Series(df.max() - df.min(), name='RANGE')
    data = pd.concat([std, range], axis=1)

    with open('dataviz.txt', 'a') as info:
        info.write('SUBJECT ' + subject.upper() + '\n')
        info.write(tabulate(data, headers=['sensor', 'std', 'range']))
        info.write('\n\n')
