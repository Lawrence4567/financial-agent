@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if /i "%SCRIPT_DIR:~0,4%"=="\\?\" set "SCRIPT_DIR=%SCRIPT_DIR:~4%"

for %%I in ("%SCRIPT_DIR%.") do set "REPO_ROOT=%%~fI"
set "PYTHON_PATH=%REPO_ROOT%\.venv\Scripts\python.exe"
set "APP_PATH=%REPO_ROOT%\01_your_canada_version\app\app_local.py"
set "DEFAULT_PORT=8506"

if not exist "%PYTHON_PATH%" (
    echo Cannot find the project virtual environment at "%PYTHON_PATH%"
    exit /b 1
)

if not exist "%APP_PATH%" (
    echo Cannot find the Streamlit app at "%APP_PATH%"
    exit /b 1
)

cd /d "%REPO_ROOT%"
set "VIRTUAL_ENV=%REPO_ROOT%\.venv"
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"
set "PYTHONNOUSERSITE=1"
"%PYTHON_PATH%" -c "import socket,sys; start=int(sys.argv[1]); port=next(p for p in range(start,start+50) if (lambda s,p: (s.connect_ex(('127.0.0.1',p))!=0))(socket.socket(),p)); print(port)" "%DEFAULT_PORT%" > "%TEMP%\canada_app_port.txt"
set /p APP_PORT=<"%TEMP%\canada_app_port.txt"

echo Starting Canada app with project venv:
echo   %PYTHON_PATH%
echo.
echo Local URL:
echo   http://localhost:%APP_PORT%
echo.

"%PYTHON_PATH%" -m streamlit run "%APP_PATH%" --server.port "%APP_PORT%" --browser.gatherUsageStats false %*
