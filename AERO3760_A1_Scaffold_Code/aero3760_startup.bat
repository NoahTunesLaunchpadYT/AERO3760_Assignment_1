@echo off

set "requirements_file=aero3760_requirements.txt"
set "env_name=aero3760_venv"

REM 1. Check for requirements file
if not exist "%requirements_file%" (
    echo ERROR: "%requirements_file%" not found in "%CD%".
    exit /b 1
)
 
REM 2. Create the virtual environment
echo Creating venv "%env_name%"...
python -m venv "%env_name%" || (
    echo ERROR: venv creation failed.
    exit /b 1
)

REM 3. Activate the virtual environment
call "%env_name%\Scripts\activate.bat"
 
REM 4. Verify activation
if not defined VIRTUAL_ENV (
    echo ERROR: Failed to activate venv.
    exit /b 1
)
 
REM 5. Force the prompt to show the venv name
prompt (%env_name%) $P$G
 
REM 6. Upgrade pip and install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r "%requirements_file%" 2>&1 | findstr /V "Requirement already satisfied"
 
echo.
echo venv is ready and activated.
echo To exit, run: deactivate
exit /b 0