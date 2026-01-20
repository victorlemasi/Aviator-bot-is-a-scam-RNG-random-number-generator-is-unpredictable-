import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Simulated or Manual Data Collection
def fetch_game_data(num_rounds=500):
    """Fetch historical Aviator game results (Simulation or Manual)."""
    choice = input("Do you want to input data manually? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("Enter multipliers one by one. Type 'done' to finish.")
        data = []
        while True:
            val = input(f"Round {len(data) + 1}: ")
            if val.lower() == 'done':
                if len(data) < 10:
                    print("Error: Need at least 10 data points.")
                    continue
                break
            try:
                multiplier = float(val)
                data.append(multiplier)
            except ValueError:
                print("Invalid input. Please enter a number.")
        return data
    else:
        print("Using simulated data...")
        return [random.uniform(1.1, 5.0) for _ in range(num_rounds)]  # Simulated multipliers

# Prepare Data for LSTM
def prepare_data(sequence, n_steps=10):
    """Convert game data into sequences for LSTM input."""
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        X.append(sequence[i:i+n_steps])
        y.append(sequence[i+n_steps])
    return np.array(X), np.array(y)

# Load and Preprocess Data
data = fetch_game_data()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

# Define Sequence Length (Past 10 rounds -> Predict next round)
n_steps = 10
X, y = prepare_data(scaled_data, n_steps)

# Reshape for LSTM Input: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
    LSTM(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(1)  # Predict next multiplier
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=16, verbose=1)

# Predict Next Multiplier
last_sequence = scaled_data[-n_steps:].reshape(1, n_steps, 1)
predicted_scaled = model.predict(last_sequence)
predicted_multiplier = scaler.inverse_transform(predicted_scaled)[0][0]

# Display Prediction
print("\n**Aviator LSTM Prediction**")
print(f"Next Multiplier: {round(predicted_multiplier, 2)}x")
print("Recommendation: Cash out at", round(predicted_multiplier * 0.9, 2), "x for safety.")

# Plot Training Loss
plt.plot(model.history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('LSTM Training Loss')
plt.show()
