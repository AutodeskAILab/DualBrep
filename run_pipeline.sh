#!/usr/bin/env bash
# DualBrep — all-in-one pipeline on the sample point clouds:
#   input -> AE decode (SDF/UDF) -> segment (clustering.py)
#         -> parametrize + latent optimisation, 24 rotations (rebuild.py)
#         -> rebuild the B-rep STEP (postprocess.py)
#
# One-time environment setup:
#   conda create -n Dualbrep python=3.12 -y
#   conda activate Dualbrep
#   conda install -c conda-forge pythonocc-core=8.0.1.1 -y
#   pip install -r requirements.txt
#
# Then download the data/checkpoint zip and extract it here so you have ./checkpoints and ./data.
#
# Override any of these with env vars:
#   CONFIG=config_pc.yaml   (config.yaml = implicit AE; config_pc.yaml = point-cloud AE)
#   OUT=output_pipeline  ROTATIONS=all  TEST_RES=256
#   NPROC=16   parametrizer processes (shards of the shapes); with NPROC > 1 they share the GPU through
#              CUDA MPS when nvidia-cuda-mps-control is available (MPS_DIR=$OUT/mps; stopped afterwards)
#   NUM_CPUS=  postprocess.py Ray CPUs (default: all cores)
#
# Usage:  ./run_pipeline.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="${CONFIG:-config_pc.yaml}"   # point-cloud AE on the bundled sample/ clouds
OUT="${OUT:-output_pipeline}"
ROTATIONS="${ROTATIONS:-all}"
TEST_RES="${TEST_RES:-256}"
NPROC="${NPROC:-16}"
NUM_CPUS="${NUM_CPUS:-}"
mkdir -p "$OUT"
MPS_DIR="${MPS_DIR:-$(cd "$OUT" && pwd)/mps}"

echo "== [1/3] reconstruct (AE) + segment  ($CONFIG, ${TEST_RES}^3) =="
python ae_reconstruct.py config="$CONFIG" runtime.test_res="$TEST_RES" \
    runtime.compute_clustering=true runtime.output_dir="$OUT/recon"

echo "== [2/3] parametrize + latent optimisation ($ROTATIONS rotations, $NPROC processes) =="
mps=0
if [ "$NPROC" -gt 1 ] && command -v nvidia-cuda-mps-control > /dev/null; then
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if nvidia-cuda-mps-control -d; then mps=1; fi
fi
pids=()
for i in $(seq 0 $((NPROC - 1))); do
    python rebuild.py --input "$OUT/recon" --out "$OUT/brep" --rotations "$ROTATIONS" \
        --shard "$i/$NPROC" > "$OUT/rebuild_$i.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do
    if ! wait "$p"; then fail=1; fi
done
if [ "$mps" = 1 ]; then echo quit | nvidia-cuda-mps-control; fi
if [ "$fail" != 0 ]; then echo "stage 2 failed, see $OUT/rebuild_*.log"; exit 1; fi

echo "== [3/3] rebuild: assemble B-rep STEP (OpenCASCADE) =="
python postprocess.py --input "$OUT/brep" ${NUM_CPUS:+--num_cpus "$NUM_CPUS"}

echo ""
echo "Done. One valid B-rep per shape: $OUT/brep/<name>.step (+ <name>.ply); candidates under $OUT/brep/tmp/"
