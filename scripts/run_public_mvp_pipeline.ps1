param(
    [int]$Epochs = 30,
    [int]$Batch = 8,
    [string]$Device = "cpu",
    [string]$Dataset = "training/datasets/public_mvp_v3",
    [string]$RunName = "public_mvp_hardneg_v3"
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
$sourceManifest = Join-Path $datasetRoot "source_manifest.json"
$runRoot = Join-Path $repoRoot ("runs/golfball/" + $RunName)
$metricsRoot = Join-Path $repoRoot ("runs/golfball/evaluation/" + $RunName)
$seedWeights = Join-Path $repoRoot "training/models/seed_golf_ball_yolov8n.pt"
$bestWeights = Join-Path $repoRoot "training/models/public_mvp_best.pt"

Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $datasetYaml -PathType Leaf)) {
        & $python training/build_public_dataset.py `
            --output $Dataset `
            --max-positives 800 `
            --max-negatives 500 `
            --scene-negatives 150
        if ($LASTEXITCODE -ne 0) { throw "Public dataset build failed" }
    }

    & $python training/validate_dataset.py `
        --data $datasetYaml `
        --manifest $datasetManifest
    if ($LASTEXITCODE -ne 0) { throw "Dataset validation failed" }

    if (Test-Path -LiteralPath $runRoot) {
        throw "Training output already exists; choose a new -RunName: $runRoot"
    }
    if (Test-Path -LiteralPath $metricsRoot) {
        throw "Evaluation output already exists; choose a new -RunName: $metricsRoot"
    }
    New-Item -ItemType Directory -Force -Path $metricsRoot | Out-Null

    & $python training/evaluate.py `
        --weights $seedWeights `
        --data $datasetYaml `
        --manifest $datasetManifest `
        --split val `
        --output (Join-Path $metricsRoot "seed-val.json") `
        --device $Device `
        --operating-threshold 0.20 `
        --recommend-thresholds
    if ($LASTEXITCODE -ne 0) { throw "Seed validation evaluation failed" }

    & $python training/train.py `
        --data $datasetYaml `
        --manifest $datasetManifest `
        --base $seedWeights `
        --epochs $Epochs `
        --imgsz 640 `
        --batch $Batch `
        --device $Device `
        --workers 0 `
        --project "runs/golfball" `
        --name $RunName `
        --seed 42
    if ($LASTEXITCODE -ne 0) { throw "Training failed" }
    & $python training/select_checkpoint.py `
        --weights-dir (Join-Path $runRoot "weights") `
        --data $datasetYaml `
        --manifest $datasetManifest `
        --output-checkpoint $bestWeights `
        --report (Join-Path $metricsRoot "checkpoint-selection.json") `
        --device $Device
    if ($LASTEXITCODE -ne 0) { throw "FP-prioritized checkpoint selection failed" }

    & $python training/evaluate.py `
        --weights $bestWeights `
        --data $datasetYaml `
        --manifest $datasetManifest `
        --split val `
        --output (Join-Path $metricsRoot "new-val.json") `
        --device $Device `
        --operating-threshold 0.20 `
        --recommend-thresholds
    if ($LASTEXITCODE -ne 0) { throw "New-model validation evaluation failed" }

    $validation = Get-Content -Raw -LiteralPath (Join-Path $metricsRoot "new-val.json") | ConvertFrom-Json
    $threshold = [double]$validation.threshold_recommendation.confirmed_average_confidence

    foreach ($model in @(
        @{ Name = "seed"; Weights = $seedWeights },
        @{ Name = "new"; Weights = $bestWeights }
    )) {
        & $python training/evaluate.py `
            --weights $model.Weights `
            --data $datasetYaml `
            --manifest $datasetManifest `
            --split test `
            --output (Join-Path $metricsRoot ($model.Name + "-test.json")) `
            --device $Device `
            --operating-threshold $threshold
        if ($LASTEXITCODE -ne 0) { throw ($model.Name + " test evaluation failed") }
    }

    & $python training/compare_offline_models.py `
        --seed (Join-Path $metricsRoot "seed-test.json") `
        --new (Join-Path $metricsRoot "new-test.json") `
        --output (Join-Path $metricsRoot "seed-vs-new.json")
    if ($LASTEXITCODE -ne 0) { throw "Offline comparison failed" }

    $checkpointHash = (Get-FileHash -LiteralPath $bestWeights -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "Best checkpoint: $bestWeights"
    Write-Host "Best checkpoint SHA256: $checkpointHash"
    Write-Host "Selected confirmed threshold: $threshold"
    Write-Host "Dataset source manifest: $sourceManifest"
    Write-Host "Comparison report: $(Join-Path $metricsRoot 'seed-vs-new.json')"
    Write-Host "Next: publish best.pt as the documented public GitHub Release asset, verify its SHA256 while signed out, set Codemagic golfballfinder_model variables, and run ios-model-compile-check."
}
finally {
    Pop-Location
}
