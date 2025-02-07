#multi-step  futur prediction(preeicting multiple values at once)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Step 1: Prepare the data  
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
x, y = [], []

#use past 3 data to predict 3 future values
for i in range(len(data) - 6):
    x.append(data[i:i+3])
    y.append(data[i+3:i+6])

x = np.array(x).reshape((len(x), 3, 1))
y = np.array(y)

# Step 2: Build the model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')

# Step 3: Train the model
model.fit(x, y, epochs=300, verbose=0)

#STEP 4: PREDICT THE NEXT 3 VALUES AFTER 80,90,100
input_data = np.array([[80, 90, 100]]).reshape(1, 3, 1)
future_pred = model.predict(input_data)
print(f"Predicted next 3 numbers after 80, 90, 100: {future_pred[0]}")