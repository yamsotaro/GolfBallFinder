param(
    [switch]$SkipSeedModel
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "GolfBallFinder Windows bootstrap" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10+ was not found on PATH. Install Python, then rerun this script."
}

$versionText = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python $versionText"
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 'Python 3.10+ is required')"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10+ is required. Found Python $versionText."
}

# Keep tool caches inside the project so bootstrap is hermetic on restricted Windows hosts.
$UltralyticsCache = Join-Path $Root ".cache\ultralytics"
$HuggingFaceCache = Join-Path $Root ".cache\huggingface"
New-Item -ItemType Directory -Force -Path $UltralyticsCache, $HuggingFaceCache | Out-Null
$env:YOLO_CONFIG_DIR = $UltralyticsCache
$env:HF_HOME = $HuggingFaceCache

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv with Python $versionText."
    }
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "The .venv Python executable is missing: $Python"
}
& $Python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "The existing .venv uses Python older than 3.10. Recreate .venv with Python 3.10+."
}
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip in .venv."
}
& $Python -m pip install -r training/requirements-windows.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install training/requirements-windows.txt."
}

if (-not $SkipSeedModel) {
    & $Python scripts/fetch_seed_model.py
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to fetch or verify the seed model."
    }
}

& $Python -m unittest discover -s Tests/Python -v
if ($LASTEXITCODE -ne 0) {
    throw "Python tests failed."
}

Write-Host ""
Write-Host "Windows bootstrap complete." -ForegroundColor Green
Write-Host "- Training/evaluation/data tools are ready in .venv."
Write-Host "- Swift/iOS source can be edited on Windows."
Write-Host "- Core ML export, Xcode compilation, signing, and TestFlight upload are delegated to Codemagic macOS CI."
Write-Host ""
Write-Host "Next: read docs/WINDOWS_CLOUD_BUILD.md and push this repository to GitHub."
