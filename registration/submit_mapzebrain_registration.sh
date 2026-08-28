#!/bin/bash
# submit_mapzebrain_registration.sh
# ==================================
# Submit a single sbatch job to register the MapZebrain GCaMP reference brain
# to the data mean brain and warp all region masks into template space.
#
# This is a one-shot job (not per-fish), so only one sbatch is submitted.
# Python path is read from config_registration.py (same as submit_registration_syn.sh).
#
# Usage (from repo root):
#     bash registration/submit_mapzebrain_registration.sh
#
# Optional flags passed through to the Python script:
#     bash registration/submit_mapzebrain_registration.sh --overwrite

SCRIPT="registration/register_mapzebrain_to_template.py"
CPUS=16
MEM="64G"
LOG_DIR="logs/registration"
LOG="${LOG_DIR}/mapzebrain_registration.log"

mkdir -p "$LOG_DIR"

# Bootstrap: use system python3 to read PYTHON_BIN from config
# (mirrors submit_registration_syn.sh exactly)
PYTHON=$(python3 -c "
import sys
sys.path.insert(0, 'registration/config')
from config_registration import PYTHON_BIN
print(PYTHON_BIN)
")

# Pass any extra CLI flags (e.g. --overwrite, --dry-run) straight through
EXTRA_ARGS="$*"

JOB_NAME="reg_mapzebrain"

echo "Submitting MapZebrain registration job..."
echo "  Script  : $SCRIPT"
echo "  Python  : $PYTHON"
echo "  CPUs    : $CPUS"
echo "  Memory  : $MEM"
echo "  Log     : $LOG"
echo "  Extras  : ${EXTRA_ARGS:-none}"

sbatch \
    --job-name="$JOB_NAME" \
    --cpus-per-task=$CPUS \
    --mem=$MEM \
    --output="$LOG" \
    --wrap="$PYTHON $SCRIPT $EXTRA_ARGS"

echo ""
echo "Job submitted."
echo "Monitor with: squeue -u \$USER"
echo "Log:          $LOG"
