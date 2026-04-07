#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Master experiment script for UNSW-NB15 Ensemble IDS
#
# Runs experiments in order:
#   1. Test suite (smoke + unit + leakage)
#   2. Binary sanity check              (3-fold, ~15 min)
#   3. Multiclass main experiment       (5-fold, ~2-3 hr)
#   4. Ablation: no_calibration         (5-fold)
#   5. Ablation: no_gate                (5-fold)
#   6. Ablation: no_feature_selection   (5-fold)
#   7. Ablation: no_engineered_features (5-fold)
#   8. Ablation: simplified_meta_features (5-fold)
#   9. Ablation: no_stability_weighting   (5-fold)
#
# Usage:
#   bash run_all.sh               # full suite
#   bash run_all.sh --skip-tests  # skip pytest, run experiments only
#   bash run_all.sh --main-only   # tests + multiclass main only (no ablations)
#
# Artifacts land in:  artifacts/<run_id>/
# Reports land in:    reports/<run_id>/
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════${RESET}"; \
            echo -e "${BOLD}${CYAN}  $*${RESET}"; \
            echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${RESET}"; }

# ── Argument parsing ───────────────────────────────────────────────────────────
SKIP_TESTS=0
MAIN_ONLY=0

for arg in "$@"; do
  case $arg in
    --skip-tests)   SKIP_TESTS=1 ;;
    --main-only)    MAIN_ONLY=1 ;;
    --help|-h)
      sed -n '3,18p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      error "Unknown argument: $arg"
      echo "Usage: bash run_all.sh [--skip-tests] [--main-only]"
      exit 1
      ;;
  esac
done

# ── Pre-flight checks ──────────────────────────────────────────────────────────
header "Pre-flight checks"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info "Working directory: $SCRIPT_DIR"

# Prefer 'python' (virtualenv) over 'python3' (system)
if command -v python &>/dev/null; then
  PYTHON=python
elif command -v python3 &>/dev/null; then
  PYTHON=python3
else
  error "Neither python nor python3 found. Activate your virtualenv first."
  exit 1
fi

PYTHON_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python: $($PYTHON --version)  (${PYTHON_VERSION})"

for f in data/UNSW_NB15_training-set.csv data/UNSW_NB15_testing-set.csv; do
  if [[ ! -f "$f" ]]; then
    error "Missing data file: $f"
    error "Ensure both official split CSVs are present in data/ before running."
    exit 1
  fi
done
success "Data files found."

# ── Tracking ───────────────────────────────────────────────────────────────────
SUITE_START=$(date +%s)
declare -A RUN_IDS
declare -A STATUSES

run_experiment() {
  local label="$1"
  local config="$2"

  header "Experiment: $label"
  info "Config: $config"
  info "Start: $(date '+%Y-%m-%d %H:%M:%S')"

  local t0
  t0=$(date +%s)

  if $PYTHON -m src.pipeline.run --config "$config"; then
    local elapsed=$(( $(date +%s) - t0 ))
    success "$label completed in $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
    STATUSES["$label"]="OK"
  else
    local elapsed=$(( $(date +%s) - t0 ))
    error "$label FAILED after $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
    STATUSES["$label"]="FAILED"
    # Continue with remaining experiments rather than aborting the full suite
  fi
}

# ── 1. Test suite ──────────────────────────────────────────────────────────────
if [[ $SKIP_TESTS -eq 0 ]]; then
  header "Test suite"
  info "Running: smoke + unit + leakage tests"
  if $PYTHON -m pytest tests/smoke/ tests/unit/ tests/leakage/ -v --tb=short -q; then
    success "All tests passed."
    STATUSES["tests"]="OK"
  else
    error "Tests failed. Fix before running experiments."
    STATUSES["tests"]="FAILED"
    exit 1
  fi
else
  warn "--skip-tests set: skipping pytest."
  STATUSES["tests"]="SKIPPED"
fi

# ── 2. Binary sanity check ─────────────────────────────────────────────────────
run_experiment "binary_sanity" "configs/binary_sanity.yaml"

# ── 3. Multiclass main ─────────────────────────────────────────────────────────
run_experiment "multiclass_main" "configs/multiclass_main.yaml"

# ── 4–9. Ablations ────────────────────────────────────────────────────────────
if [[ $MAIN_ONLY -eq 0 ]]; then
  run_experiment "ablation_no_calibration"          "configs/ablations/no_calibration.yaml"
  run_experiment "ablation_no_gate"                 "configs/ablations/no_gate.yaml"
  run_experiment "ablation_no_feature_selection"    "configs/ablations/no_feature_selection.yaml"
  run_experiment "ablation_no_engineered_features"  "configs/ablations/no_engineered_features.yaml"
  run_experiment "ablation_simplified_meta_features" "configs/ablations/simplified_meta_features.yaml"
  run_experiment "ablation_no_stability_weighting"  "configs/ablations/no_stability_weighting.yaml"
else
  warn "--main-only set: skipping ablation experiments."
  for key in ablation_no_calibration ablation_no_gate ablation_no_feature_selection \
             ablation_no_engineered_features ablation_simplified_meta_features \
             ablation_no_stability_weighting; do
    STATUSES["$key"]="SKIPPED"
  done
fi

# ── Final summary ──────────────────────────────────────────────────────────────
SUITE_END=$(date +%s)
SUITE_ELAPSED=$(( SUITE_END - SUITE_START ))

header "Run summary"
printf "%-45s  %s\n" "Experiment" "Status"
printf "%-45s  %s\n" "-----------------------------------------" "------"

ALL_OK=1
for key in tests binary_sanity multiclass_main \
           ablation_no_calibration ablation_no_gate \
           ablation_no_feature_selection ablation_no_engineered_features \
           ablation_simplified_meta_features ablation_no_stability_weighting; do
  status="${STATUSES[$key]:-SKIPPED}"
  if [[ "$status" == "OK" ]]; then
    printf "%-45s  ${GREEN}%s${RESET}\n" "$key" "$status"
  elif [[ "$status" == "FAILED" ]]; then
    printf "%-45s  ${RED}%s${RESET}\n" "$key" "$status"
    ALL_OK=0
  else
    printf "%-45s  ${YELLOW}%s${RESET}\n" "$key" "$status"
  fi
done

echo
info "Total elapsed: $(( SUITE_ELAPSED / 3600 ))h $(( (SUITE_ELAPSED % 3600) / 60 ))m $(( SUITE_ELAPSED % 60 ))s"
info "Artifacts: artifacts/"
info "Reports:   reports/"
echo

# Print the most recent run IDs from artifacts/
info "Most recent artifact directories:"
ls -td artifacts/*/  2>/dev/null | head -8 | while read -r d; do
  echo "    $d"
done

echo
if [[ $ALL_OK -eq 1 ]]; then
  success "All experiments completed successfully."
  exit 0
else
  error "One or more experiments failed. Check logs above."
  exit 1
fi
