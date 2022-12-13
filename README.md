# DigitPro
Evaluation of Machine Learning Methods to Predict Joint Angles for Finger Prosthetics

## Phase 1: 

Phase 1 implements a multi-stage dense network to perform regression and predicts future sensor values. Results for each experiment can be found in the corresponding .txt file in the folder phase1. This network is composed of two fully connected layers with a tanh activation function with a mean squared error loss.

Experiment 1: This experiment is an initial test of feasibility. Our dataset is composed of all windows of 50 time frames over subject 10 in db1. Subject 10 was arbitrarily chosen. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. The network takes as input the data from all sensors except 17 over 50 time frames and predicts the projected value for sensor 17 at the first time frame after the given input.

Experiment 2: The goal for this experiment is to determine whether the model used in Experiment 1 will perform better or worse on a subject where the data values for sensor 17 are more evenly distributed. Our dataset is composed of all windows of 50 time frames over subject 7 in db1. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. The neural network takes as input the data from all sensors except 17 over 50 time frames and predicts the projected value for sensor 17 at the first time frame after the given input. 

Experiment 3: We can reasonably derive the Distal Interphalangeal Joint from the Proximal Interphalangeal Joint, so we remove information from sensor 7 and 17 from the input. Our dataset is composed of all windows of 50 time frames over subject 10 in db1. The neural network takes as input the data from all sensors except 7 over 50 time frames and predicts the projected value for sensor 7 at the first time frame after the given input.

Experiment 4: We test if our model performs well given amputation of all fingers. We therefore only use sensors 8, 10, and 14 to predict value for sensor 5 at the first time frame after the given input. These sensors correspond to the knuckles for the middle, ring, pinky, and pointer finger, respectively.

Experiment 5: We want to determine if a model will perform well if it trained on different subjects at once. Our dataset is composed of all windows of 50 time frames over all subjects in db1. We train, test, and evaluate the network from Experiment 1 on this dataset.

Experiment 6: This goal for this experiment is to validate our expectation that, due to different hand structures among individuals, a model trained on one individual will not perform as well on others. We train the model on subject 10 with the same parameters as Experiment 1 evaluate its performance on all other subjects in db1.

Experiment 7: We compare the results from Experiment 1 with a random baseline.

## Phase 2:

Phase 2 implements a multi-stage dense network to perform regression and predict sensor values at varying moments in the future as well as varying window sizes. Results for each experiment can be found in the corresponding .txt file in the folder phase2. This network is composed of two fully connected layers with a tanh activation function with a mean squared error loss.

Experiment 1: This experiment is a test of the robustness of the model to predictions farther in the future. Our dataset is again composed of all windows of 50 time frames over subject 10 in db1. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. The neural network takes as input the data from all sensors except 17 over 50 time frames and predicts the projected value for sensor 17 at incements of 5 time steps up to 100 time steps in the future. With the data in db1 being sampled at 100 Hz, this effectively determines the ability of the model to predict at increments of 0.05 seconds, up to 1 second into the future.

Experiment 2: This experiment tests the effectiveness of various window sizes. Our dataset is composed of varying window sizes from 1, 10, 20, ..., to 100 over subject 10 in db1. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. The neural network takes as input the data from all sensors except 17 over various time frames and predicts the projected value for sensor 17 five time steps into the future. 

## Phase 3: 

Phase 3 implements a Mixture Density Network to predict a mixture of normal distributions representing predictions of future sensor values. This network has two dense layers and a third layer, the Mixture Density Layer, that gives the parameters of normal distributions. Our dataset is composed of all windows of 50 time frames over subject 10 in db1. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. The neural network takes as input the data from all sensors except 17 over 50 time frames and predicts distributions for the projected value for sensor 17 at the first time frame after the given input.

Experiment 1: We use the MDN to predict a single normal distribution for the projected value for sensor 17.

Experiment 2: This experiment expands on Experiment 1 by predicting a multiple distributions. We use the MDN to predict 5 normal distributions for the projected value for sensor 17.

Experiment 3: We compare the results of Experiment 2 with a random baseline.