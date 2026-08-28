#!/bin/bash
# submit_glm_dprime_map.sh
# ========================
# Usage (from repo root):
#     bash chemogenetic/submit/submit_glm_dprime_map.sh --config config_hcrt_trpv1_csn_120min

PYTHON="/resnick/home/ychiu/miniconda3/envs/voluseg/bin/python"
SCRIPT="chemogenetic/run/run_glm_dprime_map.py"
LOG_DIR="logs/brainmap"

CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "ERROR: --config is required."
    echo "Usage: bash chemogenetic/submit/submit_glm_dprime_map.sh --config config_hcrt_trpv1_csn_120min"
    exit 1
fi

mkdir -p "$LOG_DIR"

sbatch \
    --job-name="glm_dprime_map_${CONFIG}" \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=16G \
    --time=1:00:00 \
    --output="${LOG_DIR}/${CONFIG}.log" \
    --wrap="$PYTHON $SCRIPT --config $CONFIG"

echo "Submitted."
echo "Monitor: squeue -u \$USER"
echo "Log:     ${LOG_DIR}/${CONFIG}.log"
