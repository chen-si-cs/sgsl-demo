#!/usr/bin/env bash
set -euo pipefail
TASK=/data/users/chen/gp8001
ENV="$TASK/envs/landmark"
BASE=/data/users/chen/miniforge3/envs/ce7454-v2/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
if [ ! -x "$ENV/bin/python" ]; then
    "$BASE" -m venv --system-site-packages "$ENV"
fi
"$ENV/bin/python" -m pip install 'numpy==1.26.4' 'mediapipe==0.10.35' 'opencv-contrib-python==4.11.0.86' 'scikit-learn>=1.5,<1.8' 'imageio-ffmpeg>=0.5,<0.7' requests pillow tqdm
"$ENV/bin/python" -c 'import torch,mediapipe,numpy; print(torch.__version__,mediapipe.__version__,numpy.__version__,torch.cuda.is_available())'
"$ENV/bin/python" "$TASK/code/scripts/download_alphabet.py" /data/cold_storage/chen/gp8001/raw/asl_09az --archive-only
