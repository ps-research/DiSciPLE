#!/bin/bash
# FULL-SCALE (100% data) ablation: 9 runs = {population_density,poverty,agb} x
# {full,critic_only,base}, M=100, T=15, seed=42, NO --sample_frac. Output -> results/.
# GPU placement from smart-gpu (2026-06-04), one run per GPU, avoiding CANON's runs
# (gpu-19 + GPU0/3 of the shared nodes) and other users' GPUs:
#   gpu-36 GPU 0,1,2   gpu-14 GPU 1,2   gpu-15 GPU 1,2   gpu-22 GPU 1,2
set -u

REPO=/nfs-stor/salem.lahlou/sandeep/WACV/DiSciPLE
VENV=/nfs-stor/salem.lahlou/sandeep/WACV/envs/unsloth-venv
LOGDIR=$REPO/logs/fullscale
mkdir -p "$LOGDIR" "$REPO/results"

launch() {
    local node=$1 gpu=$2 bench=$3 variant=$4
    local outdir="results/${bench}_${variant}"
    echo "Launching $bench/$variant on $node GPU $gpu -> $outdir"
    ssh -n "$node" "source \$(conda info --base)/etc/profile.d/conda.sh && \
      conda activate $VENV && cd $REPO && \
      CUDA_VISIBLE_DEVICES=$gpu nohup setsid python -u scripts/run_experiment.py \
        --benchmark $bench --variant $variant --gpu 0 \
        --output_dir $outdir --generations 15 --population_size 100 --seed 42 \
        >| $LOGDIR/${bench}_${variant}.log 2>&1 < /dev/null &" &
    sleep 2
}

# population_density -> gpu-36 (GPU 0,1,2)
launch gpu-36 0 population_density full
launch gpu-36 1 population_density critic_only
launch gpu-36 2 population_density base
# poverty -> gpu-14 (GPU 1,2) + gpu-15 (GPU 1)
launch gpu-14 1 poverty full
launch gpu-14 2 poverty critic_only
launch gpu-15 1 poverty base
# agb -> gpu-15 (GPU 2) + gpu-22 (GPU 1,2)
launch gpu-15 2 agb full
launch gpu-22 1 agb critic_only
launch gpu-22 2 agb base

wait
echo
echo "9 FULL-SCALE (100%) runs launched -> results/  logs -> $LOGDIR/"
