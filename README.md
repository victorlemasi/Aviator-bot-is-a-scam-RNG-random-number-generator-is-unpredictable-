# Aviator Bot - LSTM Prediction

This project uses an LSTM (Long Short-Term Memory) neural network to predict valid multipliers for the Aviator game.

## ⚠️ Important for Windows Users

This project has deep file paths that conflict with Windows default path limits (260 characters). To run this project successfully, you **must** use the provided helper script or map the drive manually.

## 🚀 Quick Start

1.  **Install Python 3.12**
    - This project requires Python 3.12 or 3.11.
    - *Note: Python 3.14 is currently too new for TensorFlow.*

2.  **Run the App**
    - Simply double-click **`run_app.bat`** in this folder.
    - This script automatically handles the drive mapping and runs the bot in the correct environment.

## 🛠️ Manual Installation & Run

If you prefer to use the command line, follow these steps:

### 1. Map the Drive (Required)
To bypass the "Long Path" error, map the project folder to the `Z:` drive:

```powershell
subst Z: .
```

### 2. Install Dependencies
(Only needed once)
```powershell
Z:\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Run the Bot
```powershell
Z:\.venv\Scripts\python.exe "Version 1.1 using LSTM ALGORITHM.py"
```

## Features
- **Data Collection**: Fetch simulated data OR manually input your own.
- **Preprocessing**: Scales data for LSTM training.
- **Deep Learning**: Uses TensorFlow/Keras LSTM model.
- **Prediction**: Forecasts the next multiplier and suggests a safe cash-out point.

## ✍️ Manual Data Entry
When you run the app, you will be prompted:
- **Enter data**: Type your multipliers separated by spaces (e.g., `1.1 2.0 1.5 ...`).
  - **Requirement**: You must enter at least **11** numbers.
  - **Why?**: The AI needs 10 numbers to see a pattern + 1 to learn what comes next.
- **Simulation**: Just press **Enter** without typing anything to generate simulated data.
he app uses a 3-step process to generate the simulation and recommendation:

The Simulation (Data Generation)
Logic: It uses a standard random number generator (random.uniform(1.1, 5.0)).
What it does: It creates a list of 500 random "crash points" between 1.10x and 5.00x. This mimics a history of game rounds.
The AI Prediction (LSTM)
It takes the last 10 rounds of data (either from your manual input or the simulation).
It feeds this sequence into a Neural Network (LSTM) that has learned patterns from the previous 500 rounds.
The network outputs a single number: the Predicted Next Multiplier.
The Recommendation (Safety Margin)
The app calculates the safe cash-out point by taking 90% of the predicted value.
Formula: Recommendation = Prediction * 0.90
Reason: If the AI predicts 2.00x, it's safer to cash out slightly earlier (at 1.80x) to avoid crashing exactly at the predicted number.