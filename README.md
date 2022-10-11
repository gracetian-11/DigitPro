# DigitPro
Machine Learning for Prosthetic Finger Kinematics

Experiment 1:

Our dataset is composed of all windows of 50 time frames over the 27 subjects in db1. We randomly select 80% of these windows for training and the remaining 20% for testing.

We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. Our model should predict the information from the missing sensors. As a test of feasibility, the neural network takes as input the data from all sensors except 17 over 50 time frames and predicts the projected value for sensor 17 at the first time frame after the given input.
