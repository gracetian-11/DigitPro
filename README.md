# DigitPro
Machine Learning for Prosthetic Finger Kinematics

Experiment 1: Our dataset is composed of all windows of 50 time frames over subject 10 in db1. We randomly select 80% of these windows for training and the remaining 20% for testing. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. Our model should predict the information from the missing sensors. As a test of feasibility, the neural network takes as input the data from all sensors except 17 over 50 time frames and predicts the projected value for sensor 17 at the first time frame after the given input.

Experiment 1 Results:
test loss, test acc: [0.02604932151734829, 0.02604932151734829]
Predictions:  [-8.985901, -8.836975, -9.016388, -8.996735, -8.944824, -8.995575, -8.985321, -8.988586, -8.924774, -8.821594, -9.016602, -9.002502, -8.991394, -9.010529, -8.993408, -8.992615, -8.996735, -9.001587, -9.002197, -9.015808]
Ground Truth [-10.46429271082161, 1.415715482977248, -13.704297188234705, -7.224312903555983, -15.864291949794278, -7.224312903555983, -16.9443016656478, -2.9042987102893676, -14.784306904088226, -1.8243136645833147, -8.304297949262036, -8.304297949262036, -6.144278517554994, -14.784306904088226, -3.9842837559954205, -5.064293471848941, -7.224312903555983, 10.055694529215543, -9.384307665115557, -7.224312903555983]

Experiment 2: Due to unsatisfactory results of Experiment 1, we simplify our problem further by reducing the problem to one of classification, in which each class is defined to be a range of 10 degrees. As in Experiment 1, we use a window size of 50 time frames and hide information on sensor 17. We use a simple MLP network and train and predict the angle of sensor 17 on subject 10 in db1.

Experiment 2 Results:
