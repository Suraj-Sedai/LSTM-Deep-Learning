import tensorflow as tf
import numpy as np

# Simple data
X = np.array([1, 2, 3, 4, 5])   # Input (1, 2, 3, 4, 5)
y = np.array([2, 4, 6, 8, 10])  # Output (2, 4, 6, 8, 10)

# Building a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])  # A single neuron
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=1000, verbose=0)

# Predict the next number (for input 6)
prediction = model.predict([6])
print(f"Predicted value for 6: {prediction[0][0]}")
