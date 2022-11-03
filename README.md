# DigitPro
Machine Learning for Prosthetic Finger Kinematics

PHASE 1: Our dataset is composed of all windows of 50 time frames for a single subject. We randomly select 80% of these windows for training and the remaining 20% for testing. We remove information corresponding to a single sensor and predict the value for this sensor at the first time frame after the given input.

Experiment 1: This experiment is an initial test of feasibility. Our dataset is composed of all windows of 50 time frames over subject 10 in db1. Subject 10 was arbitrarily chosen. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. The neural network takes as input the data from all sensors except 17 over 50 time frames and predicts the projected value for sensor 17 at the first time frame after the given input. Our network is composed of two fully connected layers with a tanh activation function and the loss is computed with a mean squared error.

Experiment 2: Our dataset is composed of all windows of 50 time frames over subject 7 in db1. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. The neural network takes as input the data from all sensors except 17 over 50 time frames and predicts the projected value for sensor 17 at the first time frame after the given input. The goal for this experiment is to determine whether the model used in Experiment 1 will perform better or worse on a subject where the data values for sensor 17 are more evenly distributed.

Experiment 3: Our dataset is composed of all windows of 50 time frames over subject 7 in db1. We can reasonably derive the Distal Interphalangeal Joint from the Proximal Interphalangeal Joint, so we remove information from sensor 7 from the input. The neural network takes as input the data from all sensors except 7 over 50 time frames and predicts the projected value for sensor 7 at the first time frame after the given input. We use the same network as described in Experiment 1.

Experiment 4: Experiment 1 is rerun using sensors 8, 10, and 14 to predict value for sensor 5 at the first time frame after the given input. These sensors correspond to the knuckles for the middle, ring, pinky, and pointer finger, respectively. We use the same network as described in Experiment 1.

PHASE 1 RESULTS:
RUNNING EXPERIMENT 1...
Parsing data...
Processed db1/S10_E3_A1_angles.csv
Generating training and testing datasets...
2022-11-03 11:19:56.361216: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Compiling model...
Training model...
5355/5355 [==============================] - 6s 1ms/step - loss: 8.1054e-04 - mean_absolute_error: 0.0152
Testing model...
Generating predictions...
loss: 0.00024151828256435692
mean_absolute_error: 0.011613985523581505
  Predictions    Ground Truth       Error
-------------  --------------  ----------
   -10.6146          1.60016   12.2148
    -8.56845        -2.85513    5.71332
    -1.76865        -3.74619    1.97754
     6.13623        10.5108     4.37454
    10.6188         11.4288     0.809967
    -1.918          -1.96408    0.0460815
     0.346802        0.709106   0.362305
     1.99939         2.49124    0.491852
    -1.42245        -1.07303    0.349426
    -0.383423        5.1644     5.54782
     1.55292         0.709106   0.843811
    -2.67786        -7.31042    4.63257
    -1.85522        -2.85513    0.999908
    -0.431427        0.709106   1.14053
    -1.47986        -0.181946   1.29791
     3.32239         4.27335    0.950958
    -2.73013        -3.74619    1.01605
    -1.88812        -1.07303    0.815094
    -8.58014         5.1644    13.7445
     5.17929         7.83759    2.65829
% predictions with error > 10: 0.008730770128627122

RUNNING EXPERIMENT 2...
Parsing data...
Processed db1/S7_E3_A1_angles.csv
Generating training and testing datasets...
Compiling model...
Training model...
5534/5534 [==============================] - 7s 1ms/step - loss: 0.0013 - mean_absolute_error: 0.0218
Testing model...
Generating predictions...
loss: 0.0005297983298078179
mean_absolute_error: 0.01706608012318611
  Predictions    Ground Truth      Error
-------------  --------------  ---------
     41.945           40.4139  1.53116
     29.8555          28.8301  1.02539
     29.1202          26.1569  2.96329
     42.5378          51.1066  8.56882
      8.21301         15.5109  7.2979
     22.8027          19.9195  2.88327
     27.4058          27.4953  0.0895386
     23.4828          21.7016  1.78119
     22.5823          13.6821  8.90025
     22.2096          20.8105  1.39908
     30.7324          27.048   3.68448
     32.1452          30.6122  1.53305
     30.3383          35.0675  4.72922
     31.3851          29.7211  1.664
     29.7751          28.8301  0.945068
     45.3195          44.8692  0.450378
     18.0838          16.3552  1.72858
     34.5426          33.0609  1.48175
     42.2024          35.0675  7.13489
     27.0857          26.1569  0.928802
% predictions with error > 10: 0.03761096930131695

RUNNING EXPERIMENT 3...
Parsing data...
Processed db1/S7_E3_A1_angles.csv
Generating training and testing datasets...
Compiling model...
Training model...
5534/5534 [==============================] - 7s 1ms/step - loss: 0.0017 - mean_absolute_error: 0.0267
Testing model...
Generating predictions...
loss: 0.0006277686916291714
mean_absolute_error: 0.018257075920701027
  Predictions    Ground Truth     Error
-------------  --------------  --------
     28.1663        30.435     2.26877
     41.1445        47.7797    6.63525
     36.2814        33.1737    3.10779
     33.5928        34.0865    0.493713
     33.9285        26.3008    7.62769
     58.5316        66.3798    7.84821
     40.3109        38.6509    1.65994
      7.71849        8.52599   0.807495
     38.5346        40.4767    1.94214
     43.7393        48.3696    4.63034
     39.5083        37.7381    1.77029
     43.4943        41.3896    2.10477
     36.9765        40.4767    3.50021
     30.8375        29.5222    1.31537
     30.3001        25.8707    4.42941
      2.65414        0.310059  2.34409
     41.927         42.3024    0.375458
     28.2104        26.7835    1.42691
     34.069         35.9123    1.84326
     38.3053        39.5638    1.25854
% predictions with error > 10: 0.06263977049402517

RUNNING EXPERIMENT 4...
Parsing data...
Processed db1/S7_E3_A1_angles.csv
Generating training and testing datasets...
Compiling model...
Training model...
5534/5534 [==============================] - 6s 1ms/step - loss: 0.0213 - mean_absolute_error: 0.1006
Testing model...
Generating predictions...
loss: 0.01834595389664173
mean_absolute_error: 0.0905359536409378
  Predictions    Ground Truth      Error
-------------  --------------  ---------
     37.1282        40.2717     3.14354
     26.3704        24.065      2.30541
     39.9893        35.9123     4.07698
     10.3954        16.7419     6.34649
     43.0169        -0.521046  43.5379
     37.2209        35.9123     1.30859
     36.1699        35.9123     0.257553
     62.2587        63.8485     1.58975
     38.2034        37.7381     0.465302
     29.9835        31.8567     1.87314
     51.1968        54.1698     2.97305
     39.5952        53.6352    14.0399
     40.2834        42.3024     2.01901
     29.2889        29.5222     0.233223
      9.09398      -24.0223    33.1163
     38.643         42.3024     3.65942
     36.4308        29.2157     7.21513
     41.7505        50.5183     8.76782
     42.8947        43.8831     0.988342
     37.2912        35.079      2.2122
% predictions with error > 10: 0.2668910524294653