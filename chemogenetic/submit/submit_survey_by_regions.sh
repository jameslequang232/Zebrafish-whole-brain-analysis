#!/bin/bash
# submit_survey_by_regions.sh
# ============================
# Submits one sbatch job per config for the regional survey pipeline.
#
# Usage (from repo root):
#     bash chemogenetic/run/submit_survey_by_regions.sh
#     bash chemogenetic/run/submit_survey_by_regions.sh --overwrite
#     bash chemogenetic/run/submit_survey_by_regions.sh --top_n 20

PYTHON="/resnick/home/ychiu/miniconda3/envs/voluseg/bin/python"
SCRIPT="chemogenetic/run/run_survey_by_regions.py"
LOG_DIR="logs/chemogenetic"

EXTRA_ARGS="$*"

CONFIGS=(
    "config_hcrt_trpv1_csn_120min"
    "config_hcrt_trpv1_inj_csn_120min"
    "config_ynt185_120min"
)

mkdir -p "$LOG_DIR"

for CONFIG in "${CONFIGS[@]}"; do
    sbatch \
        --job-name="survey_by_regions_${CONFIG}" \
        --ntasks=1 \
        --cpus-per-task=25 \
        --mem=256G \
        --output="${LOG_DIR}/survey_by_regions_${CONFIG}.log" \
        --wrap="$PYTHON $SCRIPT --config $CONFIG $EXTRA_ARGS"

    echo "  Submitted: $CONFIG"
done

echo ""
echo "All ${#CONFIGS[@]} jobs submitted."
echo "Monitor: squeue -u \$USER"
echo "Logs in: $LOG_DIR/"
