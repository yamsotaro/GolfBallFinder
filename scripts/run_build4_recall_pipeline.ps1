param(
    [int]$Epochs = 15,
    [int]$Batch = 8,
    [string]$Device = "cpu",
    [string]$Dataset = "training/datasets/build4_recall_v1",
    [string]$RunName = "build4_recall_nano_v1",
    [switch]$IncludeTileSweep
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Windows environment is missing. Run scripts/bootstrap_windows.ps1 first."
}

$datasetRoot = Join-Path $repoRoot $Dataset
$datasetYaml = Join-Path $datasetRoot "dataset.yaml"
$datasetManifest = Join-Path $datasetRoot "dataset_manifest.csv"
$attribution = Join-Path $datasetRoot "attribution.csv"
$runRoot = Join-Path $repoRoot ("runs/golfball/" + $RunName)
$metricsRoot = Join-Path $repoRoot ("runs/golfball/build4_eval/" + $RunName)
$build3Weights = Join-Path $repoRoot "training/models/public_mvp_best.pt"
$selectedWeights = Join-Path $runRoot "weights/selected_build4.pt"
$selectionReport = Join-Path $metricsRoot "checkpoint-selection.json"

Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $datasetYaml -PathType Leaf)) {
        & $python training/build_recall_dataset.py `
            --source training/datasets/public_mvp_v3 `
            --output $Dataset `
            --train-synthetic 240 `
            --val-synthetic 36 `
            --test-synthetic 48 `
            --seed 404
        if ($LASTEXITCODE -ne 0) { throw "Build 4 dataset generation failed" }
    }

    & $python training/validate_dataset.py --data $datasetYaml --manifest $datasetManifest
    if ($LASTEXITCODE -ne 0) { throw "Build 4 dataset validation failed" }
    if (Test-Path -LiteralPath $runRoot) {
        throw "Training output already exists; choose a new -RunName: $runRoot"
    }
    if (Test-Path -LiteralPath $metricsRoot) {
        throw "Evaluation output already exists; choose a new -RunName: $metricsRoot"
    }
    New-Item -ItemType Directory -Force -Path $metricsRoot | Out-Null

    & $python training/train.py `
        --data $datasetYaml `
        --manifest $datasetManifest `
        --base $build3Weights `
        --epochs $Epochs `
        --imgsz 640 `
        --batch $Batch `
        --device $Device `
        --workers 0 `
        --name $RunName `
        --seed 404 `
        --patience $Epochs `
        --close-mosaic 5 `
        --mosaic 0.9 `
        --scale 0.70 `
        --translate 0.15 `
        --box 9.0 `
        --multi-scale 0.25
    if ($LASTEXITCODE -ne 0) { throw "Build 4 nano training failed" }

    & $python training/select_checkpoint.py `
        --weights-dir (Join-Path $runRoot "weights") `
        --data $datasetYaml `
        --manifest $datasetManifest `
        --output-checkpoint $selectedWeights `
        --report $selectionReport `
        --imgsz 640 `
        --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Build 4 checkpoint selection failed" }

    $selection = Get-Content -Raw -LiteralPath $selectionReport | ConvertFrom-Json
    $threshold = [double]$selection.selected.threshold_recommendation.confirmed_average_confidence
    foreach ($model in @(
        @{ Name = "build3"; Weights = $build3Weights },
        @{ Name = "build4"; Weights = $selectedWeights }
    )) {
        & $python training/evaluate.py `
            --weights $model.Weights `
            --data $datasetYaml `
            --manifest $datasetManifest `
            --attribution $attribution `
            --split test `
            --imgsz 640 `
            --device $Device `
            --operating-threshold $threshold `
            --scan-layout full `
            --output (Join-Path $metricsRoot ($model.Name + "-full-test.json"))
        if ($LASTEXITCODE -ne 0) { throw ($model.Name + " full-frame evaluation failed") }
    }

    & $python training/compare_offline_models.py `
        --seed (Join-Path $metricsRoot "build3-full-test.json") `
        --new (Join-Path $metricsRoot "build4-full-test.json") `
        --output (Join-Path $metricsRoot "build3-vs-build4.json")
    if ($LASTEXITCODE -ne 0) { throw "Build 3 versus Build 4 comparison failed" }

    if ($IncludeTileSweep) {
        foreach ($layout in @("full+five", "full+grid3")) {
            $safeName = $layout.Replace("+", "-")
            & $python training/evaluate.py `
                --weights $selectedWeights `
                --data $datasetYaml `
                --manifest $datasetManifest `
                --attribution $attribution `
                --split test `
                --imgsz 640 `
                --device $Device `
                --operating-threshold $threshold `
                --scan-layout $layout `
                --output (Join-Path $metricsRoot ("build4-" + $safeName + "-test.json"))
            if ($LASTEXITCODE -ne 0) { throw ("Build 4 " + $layout + " evaluation failed") }
        }
    }

    Write-Host "Selected checkpoint: $selectedWeights"
    Write-Host "Validation-selected confirmed threshold: $threshold"
    Write-Host "Comparison report: $(Join-Path $metricsRoot 'build3-vs-build4.json')"
    Write-Host "No Core ML export, TestFlight upload, commit, or push was performed."
}
finally {
    Pop-Location
}
