@echo off
REM Switch to the script's directory
cd /d "%~dp0"

REM Map the current folder to Z: to bypass Windows Long Path limits
REM This is necessary because the folder path is too deep for some Python libraries
subst Z: . >nul 2>&1

REM Run the app using the virtual environment on Z:
echo Running Aviator Bot...
Z:\.venv\Scripts\python.exe "Version 1.1 using LSTM ALGORITHM.py"

REM Pause so the window doesn't close immediately
pause
