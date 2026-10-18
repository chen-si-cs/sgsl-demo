#!/usr/bin/env bash
set -euo pipefail
TASK=/data/users/chen/gp8001
CODE="$TASK/code/sgsl_20261018_v1"
DATA=/data/cold_storage/chen/gp8001
PY="$TASK/envs/landmark/bin/python"
RUN="$TASK/runs/alphabet_20261018_v1"
export PYTHONPATH="$CODE/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$RUN"
for attempt in $(seq 1 240); do
    if [ -f "$DATA/raw/asl_09az/DOWNLOAD_RECEIPT.json" ]; then break; fi
    sleep 5
done
test -f "$DATA/raw/asl_09az/DOWNLOAD_RECEIPT.json"
"$PY" "$CODE/scripts/extract_alphabet.py" \
    "$DATA/raw/asl_09az/american-sign-language-09az.zip" "$DATA/landmarks/asl_09az_v1" \
    --model "$CODE/models/mediapipe/hand_landmarker.task" --workers 6 --max-per-class 1500
GPU=$("$PY" -c '
import csv, io, subprocess
def query(options):
    return list(csv.reader(io.StringIO(subprocess.check_output(["nvidia-smi", *options, "--format=csv,noheader,nounits"], text=True))))
busy = {row[0].strip() for row in query(["--query-compute-apps=gpu_uuid"])}
gpus = query(["--query-gpu=index,uuid,memory.used,utilization.gpu"])
idle = [row for row in gpus if row[1].strip() not in busy and int(row[2]) < 512 and int(row[3]) < 5]
if not idle:
    raise SystemExit("No idle GPU at training time; extraction saved for resume")
print(sorted(idle, key=lambda row: int(row[0]))[0][0].strip())
')
export CUDA_VISIBLE_DEVICES="$GPU"
echo "Selected physical GPU $GPU"
"$PY" "$CODE/scripts/train_landmark_mlp.py" "$DATA/landmarks/asl_09az_v1" "$RUN" --epochs 240
