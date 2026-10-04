@echo off
rem One-click setup and launch for Windows.
rem First run: creates a virtual environment, installs packages, gets the data,
rem trains the model (about 10-20 minutes on CPU) and opens the web app.
rem Later runs: skips straight to the web app.

cd /d "%~dp0"

rem --- pick a Python that TensorFlow supports (3.10 - 3.12) ---
set PYCMD=
for %%V in (3.11 3.10 3.12) do (
  if not defined PYCMD (
    py -%%V --version >nul 2>&1 && set PYCMD=py -%%V
  )
)
if not defined PYCMD set PYCMD=python

if not exist venv\Scripts\activate.bat (
  echo Creating virtual environment with: %PYCMD%
  %PYCMD% -m venv venv
  if errorlevel 1 (
    echo.
    echo Could not create the virtual environment. Install Python 3.10, 3.11 or 3.12 from python.org
    echo and tick "Add Python to PATH" during setup.
    pause
    exit /b 1
  )
)
call venv\Scripts\activate.bat

if not exist venv\.installed (
  echo Installing packages, this can take a few minutes...
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Package installation failed. See the Troubleshooting section in README.md.
    pause
    exit /b 1
  )
  echo done> venv\.installed
)

if not exist model\traffic_sign_model.h5 (
  if not exist model\traffic_sign_model.keras (
    echo Getting the dataset...
    python download_data.py
    echo Training the model...
    python train.py
    if errorlevel 1 (
      echo.
      echo Training failed. See the messages above and the Troubleshooting section in README.md.
      pause
      exit /b 1
    )
  )
)

if not exist samples\demo_drive.mp4 python make_samples.py

echo.
echo Starting the web app at http://127.0.0.1:5000  (close this window or press Ctrl+C to stop)
start "" cmd /c "timeout /t 5 >nul & start http://127.0.0.1:5000"
python app.py
pause
