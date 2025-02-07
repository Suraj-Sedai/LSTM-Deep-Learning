#First LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Step 1: Prepare the data
data = np.array([10, 20, 30, 40, 50, 60, 70])
X = []
y = []

for i in range(len(data) - 1):
    X.append(data[i])
    y.append(data[i + 1])

X = np.array(X).reshape((len(X), 1, 1))
y = np.array(y)

#step 2:build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#step 3: Train the model
model.fit(X, y, epochs=1000, verbose=0)

#step 4: Make Predictions
prediction = model.predict(np.array([[70]]).reshape(1, 1, 1))
print(f"Predicted next number after 70: {prediction[0][0]}")