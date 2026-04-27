$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$appPath = Join-Path $repoRoot "01_your_canada_version\\app\\app_local.py"

if (-not (Test-Path $pythonPath)) {
    throw "Cannot find the project virtual environment at $pythonPath"
}

if (-not (Test-Path $appPath)) {
    throw "Cannot find the Streamlit app at $appPath"
}

$env:VIRTUAL_ENV = Join-Path $repoRoot ".venv"
$env:PATH = (Join-Path $env:VIRTUAL_ENV "Scripts") + [IO.Path]::PathSeparator + $env:PATH
$env:PYTHONNOUSERSITE = "1"

& $pythonPath -m streamlit run $appPath --browser.gatherUsageStats false @args
