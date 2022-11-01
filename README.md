# DigitPro
Machine Learning for Prosthetic Finger Kinematics

Experiment 1: Our dataset is composed of all windows of 50 time frames over subject 10 in db1. We randomly select 80% of these windows for training and the remaining 20% for testing. We assume amputation of the tip of the second (pointer) finger so we remove information from sensor 17 from the input. Our model should predict the information from the missing sensors. As a test of feasibility, the neural network takes as input the data from all sensors except 17 over 50 time frames and predicts the projected value for sensor 17 at the first time frame after the given input.

Experiment 1 Results:
Loss, Accuracy: [0.00022828532382845879, 0.011076858267188072]
Predictions:  [-4.3919373, 8.157684, -0.97875977, 0.15118408, 5.761902, -1.3881836, 1.3059387, 17.55362, -1.3027039, 1.1553955, 8.9019165, -2.5513306, 4.350281, -0.9916382, -2.5080261, -1.0734253, -1.1257019, -0.7953186, 4.6161804, 21.47818]
Ground Truth [-3.7461853, 10.510773, -1.9640808, 0.70910645, 7.8375854, 1.6001587, -1.0730286, 18.530304, -6.4193726, 0.70910645, 7.8375854, -5.5283203, -2.855133, 4.273346, -1.9640808, -2.855133, -1.0730286, -1.9640808, 2.4912415, 13.18396]

Experiment 2: We run the same experiment as Experiment 1 on Subject 7, where the values for sensor 17 are more evenly distributed.

Experiment 2 Results: 
Loss, Accuracy: [0.000509448756929487, 0.01728430949151516]
Predictions:  [38.067886, 21.88881, 43.04738, 31.518936, 17.745499, 40.433334, 30.580795, 43.366592, 31.89473, 48.607254, 64.467545, 24.664688, 6.363922, 54.965225, -8.032837, 12.163376, 39.87892, 8.186752, 6.1349945, 31.129257]
Ground Truth [37.740677, 20.810532, 35.06749, 31.50325, 11.899933, 42.19597, 27.93901, 38.63173, 33.285385, 48.27034, 63.581436, 24.374771, 6.553543, 51.997635, -5.92128, 17.246292, 37.740677, 11.008865, 11.008865, 39.43843]

Experiment 3: Experiment 1 is rerun using sensors 8, 10, and 14 to predict value for sensor 5 at the first time frame after the given input.

Experiment 3 Results:
Loss, Accuracy: [0.018537526950240135, 0.09021467715501785]
Predictions:  [36.747402, 31.287281, 53.758823, 34.61865, 37.212612, 35.598713, 31.113941, 41.044697, 30.895962, 40.12501, 29.504436, 36.87833, 35.652065, 34.087353, 28.907436, 34.239445, 27.751644, 24.376522, 33.311466, 35.91659]
Ground Truth [34.999424, 22.219135, 51.895847, 41.38957, 37.738056, 26.783527, 36.825184, 45.953953, 31.347912, 58.671177, 39.076313, 41.38957, 28.60928, 32.26079, 22.219135, 34.08655, 30.435032, 12.177483, 36.825184, 32.26079]
