#!/bin/bash
# Autonomous pipeline: runs calibrate_lambda.py, and on success automatically
# patches main_scoring.py with the winning constants and launches the final
# 500-agent, 5-test-day confirmation run. Runs fully detached so it keeps
# working even if no one is watching the conversation.

set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate

STATUS_FILE="/tmp/calibration_pipeline_status.txt"
CALIB_LOG="/tmp/calibrate_lambda_run.log"
CONFIRM_LOG="/tmp/main_scoring_calibrated.log"

echo "$(date): pipeline started" >> "$STATUS_FILE"

caffeinate -i python calibrate_lambda.py > "$CALIB_LOG" 2>&1
CALIB_EXIT=$?

if [ "$CALIB_EXIT" -ne 0 ]; then
    echo "$(date): calibrate_lambda.py exited with code $CALIB_EXIT (likely Step 0 gate abort or crash) -- NOT patching or running confirmation. See $CALIB_LOG" >> "$STATUS_FILE"
    exit 1
fi

if [ ! -f /tmp/calibrate_lambda_final.json ]; then
    echo "$(date): calibrate_lambda.py exited 0 but no final JSON found -- NOT patching. See $CALIB_LOG" >> "$STATUS_FILE"
    exit 1
fi

echo "$(date): calibrate_lambda.py finished successfully, applying calibration" >> "$STATUS_FILE"
python apply_calibration.py >> "$STATUS_FILE" 2>&1
PATCH_EXIT=$?

if [ "$PATCH_EXIT" -ne 0 ]; then
    echo "$(date): apply_calibration.py failed (exit $PATCH_EXIT) -- NOT running confirmation" >> "$STATUS_FILE"
    exit 1
fi

echo "$(date): patch applied, launching final 500-agent 5-test-day confirmation run" >> "$STATUS_FILE"
caffeinate -i python main_scoring.py > "$CONFIRM_LOG" 2>&1
CONFIRM_EXIT=$?

if grep -q "CEVD mean requests served" "$CONFIRM_LOG" 2>/dev/null; then
    echo "$(date): confirmation run finished cleanly. Final result in $CONFIRM_LOG" >> "$STATUS_FILE"
else
    echo "$(date): confirmation run exited (code $CONFIRM_EXIT) without the expected completion marker -- check $CONFIRM_LOG" >> "$STATUS_FILE"
fi

echo "$(date): pipeline done" >> "$STATUS_FILE"
