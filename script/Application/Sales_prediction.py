# Step 7: Visualizing the Sales Trendimport numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load Sample Sales Data
data = {
    "Month": pd.date_range(start="2020-01-01", periods=36, freq="M"),
    "Sales": [100, 220, 250, 280, 300, 320, 400, 450, 500, 550, 600, 700,
              150, 170, 190, 210, 230, 250, 300, 350, 380, 400, 420, 500,
              220, 240, 260, 280, 300, 320, 350, 380, 400, 420, 440, 500]
}
df = pd.DataFrame(data)

# Step 2: Preprocess Data
sales = df['Sales'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0,1))
sales_scaled = scaler.fit_transform(sales)

# Step 3: Prepare Training Data (Use Past 12 Months to Predict Next Month)
X_train, y_train = [], []
time_steps = 12

for i in range(len(sales_scaled) - time_steps):
    X_train.append(sales_scaled[i:i+time_steps])  # Past 12 months
    y_train.append(sales_scaled[i+time_steps])  # Predict next month

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Step 4: Build LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mse')

# Step 5: Train Model
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

# Step 6: Predict Next Month's Sales
last_12_months = sales_scaled[-time_steps:].reshape(1, time_steps, 1)
predicted_sales = model.predict(last_12_months)
predicted_sales = scaler.inverse_transform(predicted_sales)  # Convert back to original scale

print(f"Predicted Sales for Next Month: {predicted_sales[0][0]}")

# Step 7: Visualizing Sales Prediction
plt.plot(df['Month'], df['Sales'], label="Actual Sales")
plt.axhline(predicted_sales[0][0], color='red', linestyle='dashed', label="Predicted Sales")
plt.legend()
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

plt.plot(df.index, df['Sales'], label="Actual Sales")
plt.axhline(predicted_sales[0][0], color='red', linestyle='dashed', label="Predicted Sales")
plt.legend()
plt.show()
