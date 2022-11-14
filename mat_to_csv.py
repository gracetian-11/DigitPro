import numpy as np
import h5py
import scipy.io
import pandas as pd
import os


def getAngles(floc, DB, name): 
    mat = scipy.io.loadmat(floc)
    angles = pd.DataFrame(mat['angles'])
    angles.to_csv("dataset_csv/" + DB + "/" + name + "_angles.csv")

DB_num = "" 
E_num = ""

if not os.path.exists("dataset_csv"):
    
    #create directories for csv files 
    os.mkdir("dataset_csv")
    os.mkdir("dataset_csv/DB1")
    os.mkdir("dataset_csv/DB2")
    os.mkdir("dataset_csv/DB5")

    for DB_dir in os.listdir('dataset'):
        if DB_dir == "DB1": 
            DB_num = "DB1"
            E_num = "E3"
        elif DB_dir == "DB2":
            DB_num = "DB2"
            E_num = "E2"
        elif DB_dir == "DB5": 
            DB_num = "DB5"
            E_num = "E3"
        else: 
            continue

        for s_dir in os.listdir("dataset" + "/" + DB_dir): 
                if s_dir != ".DS_Store":
                    for data_file in os.listdir("dataset" + "/" + DB_dir + "/" + s_dir):
                        if data_file.find(E_num) != -1: 
                            getAngles("dataset/" + DB_dir + "/" + s_dir + "/" + data_file, DB_num, data_file[:-4])
        
        """""
        if DB_dir == "DB1": 
            for s_dir in os.listdir("dataset" + "/" + DB_dir): 
                if s_dir != ".DS_Store":
                    for data_file in os.listdir("dataset" + "/" + DB_dir + "/" + s_dir):
                        if data_file.find("E3") != -1: 
                            getAngles("dataset/" + DB_dir + "/" + s_dir + "/" + data_file, "DB1", data_file[:-4])
        elif DB_dir == "DB2": 
            for s_dir in os.listdir("dataset" + "/" + DB_dir): 
                if s_dir != ".DS_Store":
                    for data_file in os.listdir("dataset" + "/" + DB_dir + "/" + s_dir):
                        if data_file.find("E2") != -1: 
                            getAngles("dataset/" + DB_dir + "/" + s_dir + "/" + data_file, "DB2", data_file[:-4])
        elif DB_dir == "DB5": 
            for s_dir in os.listdir("dataset" + "/" + DB_dir): 
                if s_dir != ".DS_Store":
                    for data_file in os.listdir("dataset" + "/" + DB_dir + "/" + s_dir):
                        if data_file.find("E3") != -1: 
                            getAngles("dataset/" + DB_dir + "/" + s_dir + "/" + data_file, "DB5", data_file[:-4])
        """