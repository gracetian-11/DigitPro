import data
import tensorflow as tf
import normalize

LABEL = 14  # sensor 17 corresponds to index 15 after dropping sensor 6 due to missing information
WINDOW_SIZE = 50  # number of time frames for one prediction
NUM_EPOCHS = 1  # specify number of epochs to train over
BATCH_SIZE = 32  # specify batch size 

file = 'db1/S10_E3_A1_angles.csv'
dataset = data.Data([file], WINDOW_SIZE, [LABEL])

# define model for experiment 1
model = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='tanh'),
    tf.keras.layers.Dense(units=32, activation='tanh'),
    tf.keras.layers.Dense(units=1, activation='tanh'),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

# compile the model with L1 loss and Adam optimizer
print("Compiling model...")
model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])
print("Compiling model... done! :)")
print("Training model...")
history = model.fit(dataset.training_features, dataset.training_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
print("Training model... done! :)")

# test model
print("Testing model...")
results = model.evaluate(dataset.testing_features, dataset.testing_labels, batch_size=BATCH_SIZE, verbose=0)
print("Testing model... done! :)")

# display predictions vs ground truth for 20 windows
print("Generating predictions...")
print("Loss, Accuracy:", results)
predictions = model.predict(dataset.testing_features, verbose=0)
unscaled_predictions = []
unscaled_testing_labels = [] 
for p in range(len(predictions)):
    norm_vals = dataset.file_norm_vals[dataset.testing_files[p]]
    unscaled_predictions.append(normalize.unscale_from_range(predictions[p], norm_vals[0], norm_vals[1], -1, 1))
    unscaled_testing_labels.append(normalize.unscale_from_range(dataset.testing_labels[p], norm_vals[0], norm_vals[1], -1, 1))
print("Predictions: ", [p[0][0] for p in unscaled_predictions[:20]])
print("Ground Truth", [l[0] for l in unscaled_testing_labels[:20]])
print("Generating predictions... done! :)")
