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
When you run the app, ask if you want to input data manually:
- Type **`y`** to enter your own list of multipliers (type `done` to finish).
- Type **`n`** (or just press Enter) to run with simulated data.
