import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Step 1: Generate Fake Sales Data (for simplicity)
dates = pd.date_range(start="2021-01-01", periods=500, freq='D')  # 500 days of data
sales = np.cumsum(np.random.randint(5, 15, size=len(dates)))  # Increasing sales trend

df = pd.DataFrame({"Date": dates, "Sales": sales})
df.set_index("Date", inplace=True)

# Step 2: Normalize Sales Data
scaler = MinMaxScaler(feature_range=(0,1))
sales_scaled = scaler.fit_transform(df['Sales'].values.reshape(-1,1))

# Step 3: Prepare Training Data
X_train, y_train = [], []
time_steps = 30  # Use past 30 days to predict the next day

for i in range(len(sales_scaled) - time_steps):
    X_train.append(sales_scaled[i:i+time_steps])
    y_train.append(sales_scaled[i+time_steps])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Reshape for LSTM

# Step 4: Build the LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Step 5: Train Model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Step 6: Predict the Next Day's Sales
last_30_days = sales_scaled[-time_steps:].reshape(1, time_steps, 1)
predicted_sales = model.predict(last_30_days)

# Convert back to original scale
predicted_sales = scaler.inverse_transform(predicted_sales)
print(f"Predicted Sales for Next Day: {predicted_sales[0][0]}")

# Step 7: Visualizing the Sales Trend
plt.plot(df.index, df['Sales'], label="Actual Sales")
plt.axhline(predicted_sales[0][0], color='red', linestyle='dashed', label="Predicted Sales")
plt.legend()
plt.show()
