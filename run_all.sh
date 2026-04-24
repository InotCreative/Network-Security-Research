#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Master experiment script for UNSW-NB15 Ensemble IDS
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
#   bash run_all.sh               # full UNSW-NB15 suite
#   bash run_all.sh --skip-tests  # skip pytest, run experiments only
#   bash run_all.sh --main-only   # tests + multiclass main only (no ablations)
#   bash run_all.sh --with-cic    # also run CIC-IDS2017 generalisation experiment
#                                 #   (requires data/CIC-IDS2017/*.csv)
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
WITH_CIC=0

for arg in "$@"; do
  case $arg in
    --skip-tests)   SKIP_TESTS=1 ;;
    --main-only)    MAIN_ONLY=1 ;;
    --with-cic)     WITH_CIC=1 ;;
    --help|-h)
      sed -n '3,18p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      error "Unknown argument: $arg"
      echo "Usage: bash run_all.sh [--skip-tests] [--main-only] [--with-cic]"
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

# CIC-IDS2017 pre-flight (only when --with-cic was requested)
CIC_DATA_PATH="data/CIC-IDS2017"
if [[ $WITH_CIC -eq 1 ]]; then
  if [[ ! -d "$CIC_DATA_PATH" ]] && [[ ! -f "$CIC_DATA_PATH" ]]; then
    error "--with-cic set but '$CIC_DATA_PATH' not found."
    error "Place the CIC-IDS2017 CSVs under data/CIC-IDS2017/ (or a single merged CSV at that path)."
    exit 1
  fi
  success "CIC-IDS2017 data path found: $CIC_DATA_PATH"
fi

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

# ── 3b. CIC-IDS2017 generalisation (optional) ──────────────────────────────────
if [[ $WITH_CIC -eq 1 ]]; then
  run_experiment "multiclass_main_cic" "configs/multiclass_main_cic.yaml"
else
  STATUSES["multiclass_main_cic"]="SKIPPED"
fi

# ── 4–12. Ablations ───────────────────────────────────────────────────────────
if [[ $MAIN_ONLY -eq 0 ]]; then
  # Component ablations (disable one novel contribution at a time)
  run_experiment "ablation_no_calibration"           "configs/ablations/no_calibration.yaml"
  run_experiment "ablation_no_feature_selection"     "configs/ablations/no_feature_selection.yaml"
  run_experiment "ablation_no_engineered_features"   "configs/ablations/no_engineered_features.yaml"
  run_experiment "ablation_simplified_meta_features" "configs/ablations/simplified_meta_features.yaml"
  run_experiment "ablation_no_stability_weighting"   "configs/ablations/no_stability_weighting.yaml"
  run_experiment "ablation_single_selector"          "configs/ablations/single_selector.yaml"

  # Path-mixing ablations (answer: which β policy is best?)
  run_experiment "ablation_stacker_only"             "configs/ablations/stacker_only.yaml"
  run_experiment "ablation_weighted_avg_only"        "configs/ablations/weighted_avg_only.yaml"
  run_experiment "ablation_no_gate"                  "configs/ablations/no_gate.yaml"
else
  warn "--main-only set: skipping ablation experiments."
  for key in ablation_no_calibration ablation_no_feature_selection \
             ablation_no_engineered_features ablation_simplified_meta_features \
             ablation_no_stability_weighting ablation_single_selector \
             ablation_stacker_only ablation_weighted_avg_only ablation_no_gate; do
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
for key in tests binary_sanity multiclass_main multiclass_main_cic \
           ablation_no_calibration \
           ablation_no_feature_selection ablation_no_engineered_features \
           ablation_simplified_meta_features ablation_no_stability_weighting \
           ablation_single_selector \
           ablation_stacker_only ablation_weighted_avg_only ablation_no_gate; do
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

# ── Ablation aggregation ───────────────────────────────────────────────────────
# Produces the required ablation_results.csv + comparison plot from CLAUDE.md.
# Runs even if some experiments failed (reads whatever results exist).
header "Ablation aggregation"
if $PYTHON -m src.eval.aggregate_ablations \
      --artifacts-dir artifacts \
      --output-dir reports; then
  success "ablation_results.csv written to reports/"
  STATUSES["aggregate"]="OK"
else
  warn "Ablation aggregation produced warnings (some experiments may not have run yet)."
  STATUSES["aggregate"]="WARN"
fi

echo
if [[ $ALL_OK -eq 1 ]]; then
  success "All experiments completed successfully."
  exit 0
else
  error "One or more experiments failed. Check logs above."
  exit 1
fi
