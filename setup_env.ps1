param(
    [string]$CondaExe = "D:\dev\anaconda\Scripts\conda.exe",
    [string]$EnvironmentPath = "D:\dev\anacondalenvs\26xv01q",
    [switch]$InstallGpu
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $EnvironmentPath "python.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$GpuRequirementsPath = Join-Path $ProjectRoot "requirements-gpu-cuda13.txt"

if (-not (Test-Path -LiteralPath $CondaExe)) {
    throw "Conda executable not found: $CondaExe"
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    & $CondaExe create --prefix $EnvironmentPath python=3.11 pip -y
    if ($LASTEXITCODE -ne 0) {
        throw "Conda environment creation failed."
    }
}

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r $RequirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "Python dependency installation failed."
}

if ($InstallGpu) {
    & $PythonExe -m pip install -r $GpuRequirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "GPU dependency installation failed."
    }
}

Write-Host "Environment ready: $EnvironmentPath"
Write-Host "Start desktop workspace: $PythonExe $ProjectRoot\run_pycharm.py"
Write-Host "Start legacy web workspace: $PythonExe $ProjectRoot\start_app.py"
Write-Host "Start CAD analysis directly: $PythonExe $ProjectRoot\media_web_app.py"
