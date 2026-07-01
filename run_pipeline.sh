#!/usr/bin/env bash
# DualBrep — all-in-one pipeline on one sample:
#   input -> AE/VAE decode (SDF/UDF) -> segment -> parametrize (UV grids) -> rebuild (B-rep STEP)
#
# One-time environment setup:
#   conda create -n Dualbrep python=3.12 -y
#   conda activate Dualbrep
#   conda install -c conda-forge pythonocc-core=7.9.0 -y
#   pip install -r requirements.txt
#
# Then download the data/checkpoint zip and extract it here so you have ./checkpoints and ./data.
#
# Override any of these with env vars:
#   CONFIG=config.yaml   (config.yaml = implicit AE; config_pc.yaml = point-cloud AE)
#   OUT=output_pipeline  ROTATIONS=all  TEST_RES=256
#
# Usage:  ./run_pipeline.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="${CONFIG:-config_pc.yaml}"   # point-cloud AE on the bundled sample/ clouds
OUT="${OUT:-output_pipeline}"
ROTATIONS="${ROTATIONS:-all}"
TEST_RES="${TEST_RES:-256}"

echo "== [1/3] reconstruct (AE) + segment  ($CONFIG, ${TEST_RES}^3) =="
python ae_reconstruct.py config="$CONFIG" runtime.test_res="$TEST_RES" \
    runtime.compute_clustering=true runtime.output_dir="$OUT/recon"

echo "== [2/3] parametrize: per-face UV grids + intersection edges + topology ($ROTATIONS rotations) =="
python rebuild.py --input "$OUT/recon" --out "$OUT/brep" --rotations "$ROTATIONS"

echo "== [3/3] rebuild: assemble B-rep STEP (OpenCASCADE) =="
python postprocess.py --input "$OUT/brep"

echo ""
echo "Done. One valid B-rep per shape: $OUT/brep/<name>.step (+ <name>.ply); candidates under $OUT/brep/tmp/"
