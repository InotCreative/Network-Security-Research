#!/usr/bin/env pwsh
# =============================================================================
# run_all.ps1 — Master experiment script for UNSW-NB15 Ensemble IDS
#
# Runs experiments in order:
#   1.  Test suite (smoke + unit + leakage)
#   2.  Binary sanity check                    (3-fold, ~15 min)
#   3.  Multiclass main experiment             (5-fold, ~2-3 hr)
#   4.  Ablation: no_calibration              (5-fold)
#   5.  Ablation: no_gate                     (5-fold, β=0.5 fixed)
#   6.  Ablation: stacker_only                (5-fold, β=1.0 fixed)
#   7.  Ablation: weighted_avg_only           (5-fold, β=0.0 fixed)
#   8.  Ablation: no_feature_selection        (5-fold)
#   9.  Ablation: no_engineered_features      (5-fold)
#   10. Ablation: simplified_meta_features    (5-fold)
#   11. Ablation: no_stability_weighting      (5-fold)
#   12. Ablation: single_selector             (5-fold, mutual_info only)
#
# Ablations 5–7 together answer: "Which mixing strategy is best?"
#   β=1.0 (stacker only) vs β=0.5 (equal) vs β=0.0 (WA only) vs learned β.
#
# Ablation 12 answers: "Does consensus of 4 selectors beat one selector?"
#
# Usage:
#   pwsh run_all.ps1               # full UNSW-NB15 suite
#   pwsh run_all.ps1 -SkipTests    # skip pytest, run experiments only
#   pwsh run_all.ps1 -MainOnly     # tests + multiclass main only (no ablations)
#   pwsh run_all.ps1 -WithCic      # also run CIC-IDS2017 generalisation experiment
#                                  #   (requires data/CIC-IDS2017/*.csv)
#
# Artifacts land in:  artifacts/<run_id>/
# Reports land in:    reports/<run_id>/
# =============================================================================

param(
    [switch]$SkipTests,
    [switch]$MainOnly,
    [switch]$WithCic,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# ── Colour helpers ─────────────────────────────────────────────────────────────
function Write-Info    { Write-Host "[INFO]  " -ForegroundColor Cyan -NoNewline; Write-Host $args }
function Write-Success { Write-Host "[OK]    " -ForegroundColor Green -NoNewline; Write-Host $args }
function Write-Warn    { Write-Host "[WARN]  " -ForegroundColor Yellow -NoNewline; Write-Host $args }
function Write-Error-Custom { Write-Host "[ERROR] " -ForegroundColor Red -NoNewline; Write-Host $args }
function Write-Header {
    Write-Host ""
    Write-Host "══════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  $args" -ForegroundColor Cyan
    Write-Host "══════════════════════════════════════════════" -ForegroundColor Cyan
}

# ── Argument parsing ───────────────────────────────────────────────────────────
if ($Help) {
    Get-Content $PSCommandPath | Select-Object -Skip 2 -First 16 | ForEach-Object { $_ -replace '^# ?' }
    exit 0
}

# ── Pre-flight checks ──────────────────────────────────────────────────────────
Write-Header "Pre-flight checks"

$SCRIPT_DIR = Split-Path -Parent $PSCommandPath
Set-Location $SCRIPT_DIR

Write-Info "Working directory: $SCRIPT_DIR"

# Prefer 'python' (virtualenv) over 'python3' (system)
$PYTHON = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $PYTHON = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PYTHON = "python3"
} else {
    Write-Error-Custom "Neither python nor python3 found. Activate your virtualenv first."
    exit 1
}

$PYTHON_VERSION = & $PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Info "Python: $(& $PYTHON --version)  ($PYTHON_VERSION)"

$dataFiles = @("data/UNSW_NB15_training-set.csv", "data/UNSW_NB15_testing-set.csv")
foreach ($f in $dataFiles) {
    if (-not (Test-Path $f)) {
        Write-Error-Custom "Missing data file: $f"
        Write-Error-Custom "Ensure both official split CSVs are present in data/ before running."
        exit 1
    }
}
Write-Success "Data files found."

# CIC-IDS2017 pre-flight (only when -WithCic was requested)
$CIC_DATA_PATH = "data/CIC-IDS2017"
if ($WithCic) {
    if (-not (Test-Path $CIC_DATA_PATH)) {
        Write-Error-Custom "-WithCic set but '$CIC_DATA_PATH' not found."
        Write-Error-Custom "Place the CIC-IDS2017 CSVs under data/CIC-IDS2017/ (or a single merged CSV at that path)."
        exit 1
    }
    Write-Success "CIC-IDS2017 data path found: $CIC_DATA_PATH"
}

# ── Tracking ───────────────────────────────────────────────────────────────────
$SUITE_START = Get-Date
$STATUSES = @{}

function Run-Experiment {
    param(
        [string]$Label,
        [string]$Config
    )

    Write-Header "Experiment: $Label"
    Write-Info "Config: $Config"
    Write-Info "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    $t0 = Get-Date

    try {
        & $PYTHON -m src.pipeline.run --config $Config
        if ($LASTEXITCODE -eq 0) {
            $elapsed = (Get-Date) - $t0
            $minutes = [math]::Floor($elapsed.TotalMinutes)
            $seconds = $elapsed.Seconds
            Write-Success "$Label completed in ${minutes}m ${seconds}s"
            $STATUSES[$Label] = "OK"
        } else {
            throw "Process exited with code $LASTEXITCODE"
        }
    } catch {
        $elapsed = (Get-Date) - $t0
        $minutes = [math]::Floor($elapsed.TotalMinutes)
        $seconds = $elapsed.Seconds
        Write-Error-Custom "$Label FAILED after ${minutes}m ${seconds}s"
        $STATUSES[$Label] = "FAILED"
        # Continue with remaining experiments rather than aborting the full suite
    }
}

# ── 1. Test suite ──────────────────────────────────────────────────────────────
if (-not $SkipTests) {
    Write-Header "Test suite"
    Write-Info "Running: smoke + unit + leakage tests"
    try {
        & $PYTHON -m pytest tests/smoke/ tests/unit/ tests/leakage/ -v --tb=short -q
        if ($LASTEXITCODE -eq 0) {
            Write-Success "All tests passed."
            $STATUSES["tests"] = "OK"
        } else {
            throw "Tests failed"
        }
    } catch {
        Write-Error-Custom "Tests failed. Fix before running experiments."
        $STATUSES["tests"] = "FAILED"
        exit 1
    }
} else {
    Write-Warn "-SkipTests set: skipping pytest."
    $STATUSES["tests"] = "SKIPPED"
}

# ── 2. Binary sanity check ─────────────────────────────────────────────────────
Run-Experiment "binary_sanity" "configs/binary_sanity.yaml"

# ── 3. Multiclass main ─────────────────────────────────────────────────────────
Run-Experiment "multiclass_main" "configs/multiclass_main.yaml"

# ── 3b. CIC-IDS2017 generalisation (optional) ──────────────────────────────────
if ($WithCic) {
    Run-Experiment "multiclass_main_cic" "configs/multiclass_main_cic.yaml"
} else {
    $STATUSES["multiclass_main_cic"] = "SKIPPED"
}

# ── 4–12. Ablations ───────────────────────────────────────────────────────────
if (-not $MainOnly) {
    # Component ablations (disable one novel contribution at a time)
    Run-Experiment "ablation_no_calibration"           "configs/ablations/no_calibration.yaml"
    Run-Experiment "ablation_no_feature_selection"     "configs/ablations/no_feature_selection.yaml"
    Run-Experiment "ablation_no_engineered_features"   "configs/ablations/no_engineered_features.yaml"
    Run-Experiment "ablation_simplified_meta_features" "configs/ablations/simplified_meta_features.yaml"
    Run-Experiment "ablation_no_stability_weighting"   "configs/ablations/no_stability_weighting.yaml"
    Run-Experiment "ablation_single_selector"          "configs/ablations/single_selector.yaml"

    # Path-mixing ablations (answer: which β policy is best?)
    Run-Experiment "ablation_stacker_only"             "configs/ablations/stacker_only.yaml"
    Run-Experiment "ablation_weighted_avg_only"        "configs/ablations/weighted_avg_only.yaml"
    Run-Experiment "ablation_no_gate"                  "configs/ablations/no_gate.yaml"
} else {
    Write-Warn "-MainOnly set: skipping ablation experiments."
    $ablationKeys = @(
        "ablation_no_calibration", "ablation_no_feature_selection",
        "ablation_no_engineered_features", "ablation_simplified_meta_features",
        "ablation_no_stability_weighting", "ablation_single_selector",
        "ablation_stacker_only", "ablation_weighted_avg_only", "ablation_no_gate"
    )
    foreach ($key in $ablationKeys) {
        $STATUSES[$key] = "SKIPPED"
    }
}

# ── Final summary ──────────────────────────────────────────────────────────────
$SUITE_END = Get-Date
$SUITE_ELAPSED = $SUITE_END - $SUITE_START

Write-Header "Run summary"
Write-Host ("{0,-45}  {1}" -f "Experiment", "Status")
Write-Host ("{0,-45}  {1}" -f "-----------------------------------------", "------")

$ALL_OK = $true
$experimentKeys = @(
    "tests", "binary_sanity", "multiclass_main", "multiclass_main_cic",
    "ablation_no_calibration",
    "ablation_no_feature_selection", "ablation_no_engineered_features",
    "ablation_simplified_meta_features", "ablation_no_stability_weighting",
    "ablation_single_selector",
    "ablation_stacker_only", "ablation_weighted_avg_only", "ablation_no_gate"
)

foreach ($key in $experimentKeys) {
    $status = if ($STATUSES.ContainsKey($key)) { $STATUSES[$key] } else { "SKIPPED" }
    if ($status -eq "OK") {
        Write-Host ("{0,-45}  " -f $key) -NoNewline
        Write-Host $status -ForegroundColor Green
    } elseif ($status -eq "FAILED") {
        Write-Host ("{0,-45}  " -f $key) -NoNewline
        Write-Host $status -ForegroundColor Red
        $ALL_OK = $false
    } else {
        Write-Host ("{0,-45}  " -f $key) -NoNewline
        Write-Host $status -ForegroundColor Yellow
    }
}

Write-Host ""
$hours = [math]::Floor($SUITE_ELAPSED.TotalHours)
$minutes = $SUITE_ELAPSED.Minutes
$seconds = $SUITE_ELAPSED.Seconds
Write-Info "Total elapsed: ${hours}h ${minutes}m ${seconds}s"
Write-Info "Artifacts: artifacts/"
Write-Info "Reports:   reports/"
Write-Host ""

# Print the most recent run IDs from artifacts/
Write-Info "Most recent artifact directories:"
if (Test-Path "artifacts") {
    Get-ChildItem -Path "artifacts" -Directory | 
        Sort-Object LastWriteTime -Descending | 
        Select-Object -First 8 | 
        ForEach-Object { Write-Host "    $($_.FullName)" }
}

# ── Ablation aggregation ───────────────────────────────────────────────────────
# Produces the required ablation_results.csv + comparison plot from CLAUDE.md.
# Runs even if some experiments failed (reads whatever results exist).
Write-Header "Ablation aggregation"
try {
    & $PYTHON -m src.eval.aggregate_ablations --artifacts-dir artifacts --output-dir reports
    if ($LASTEXITCODE -eq 0) {
        Write-Success "ablation_results.csv written to reports/"
        $STATUSES["aggregate"] = "OK"
    } else {
        throw "Process exited with code $LASTEXITCODE"
    }
} catch {
    Write-Warn "Ablation aggregation produced warnings (some experiments may not have run yet)."
    $STATUSES["aggregate"] = "WARN"
}

Write-Host ""
if ($ALL_OK) {
    Write-Success "All experiments completed successfully."
    exit 0
} else {
    Write-Error-Custom "One or more experiments failed. Check logs above."
    exit 1
}
