# DigitPro
Machine Learning for Prosthetic Finger Kinematics

## PHASE 1 (Phil): 

Our dataset is composed of all windows of 50 time frames for a single subject. We randomly select 80% of these windows for training and the remaining 20% for testing. We remove information corresponding to a single sensor and predict the value for this sensor at the first time frame after the given input.

Experiment 1: This experiment is an initial test of feasibility. Our dataset is composed of all windows of 50 time frames over subject 10 in db1. Subject 10 was arbitrarily chosen. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. The neural network takes as input the data from all sensors except 17 over 50 time frames and predicts the projected value for sensor 17 at the first time frame after the given input. Our network is composed of two fully connected layers with a tanh activation function and the loss is computed with a mean squared error.

Experiment 2: Our dataset is composed of all windows of 50 time frames over subject 7 in db1. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. The neural network takes as input the data from all sensors except 17 over 50 time frames and predicts the projected value for sensor 17 at the first time frame after the given input. The goal for this experiment is to determine whether the model used in Experiment 1 will perform better or worse on a subject where the data values for sensor 17 are more evenly distributed.

Experiment 3: Our dataset is composed of all windows of 50 time frames over subject 7 in db1. We can reasonably derive the Distal Interphalangeal Joint from the Proximal Interphalangeal Joint, so we remove information from sensor 7 from the input. The neural network takes as input the data from all sensors except 7 over 50 time frames and predicts the projected value for sensor 7 at the first time frame after the given input. We use the same network as described in Experiment 1.

Experiment 4: Experiment 1 is rerun using sensors 8, 10, and 14 to predict value for sensor 5 at the first time frame after the given input. These sensors correspond to the knuckles for the middle, ring, pinky, and pointer finger, respectively. We use the same network as described in Experiment 1.

## Phase 2 (Charlie):

Experiment 1: This experiment is a test of the robustness of the model to predictions farther in the future. Our dataset is again composed of all windows of 50 time frames over subject 10 in db1. Subject 10 was arbitrarily chosen. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. The neural network takes as input the data from all sensors except 17 over 50 time frames and predicts the projected value for sensor 17 at incements of 5 time steps up to 100 time steps in the future. With the data in db1 being sampled at 100 Hz, this effectively determines the ability of the model to predict at increments of 0.05s, up to 1s into the future. Our network is composed of two fully connected layers with a tanh activation function and the loss is computed with a mean squared error.