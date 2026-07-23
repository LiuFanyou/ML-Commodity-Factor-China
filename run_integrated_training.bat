@echo off
setlocal
cd /d "%~dp0"

set "FUTURES_DATA_ROOT=%~dp0"
set "FUTURES_INTEGRATED_CSV=%~dp0futures_contract_daily_30varieties_2015_2025.csv"
set "FUTURES_FEATURE_MODE=integrated"

if not exist "futures_env\Scripts\python.exe" (
  echo [ERROR] Missing futures_env\Scripts\python.exe
  echo Create the environment and install requirements_futures_ml.txt first.
  pause
  exit /b 1
)

if not exist "%FUTURES_INTEGRATED_CSV%" (
  echo [ERROR] Missing %FUTURES_INTEGRATED_CSV%
  pause
  exit /b 1
)

echo [INFO] Starting integrated-data training. This full 2015-2025 run can take a long time.
"futures_env\Scripts\python.exe" -u "china_futures_ml_kaggle.py" 1>"integrated_training.log" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Training failed with exit code %EXIT_CODE%.
  echo See integrated_training.log for the traceback.
  powershell -NoProfile -Command "Get-Content -LiteralPath 'integrated_training.log' -Tail 40"
  pause
  exit /b %EXIT_CODE%
)

echo [DONE] Training completed. Results are in futures_ml_outputs.
echo [INFO] Full log: integrated_training.log
pause
