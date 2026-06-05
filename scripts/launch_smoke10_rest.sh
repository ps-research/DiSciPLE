#!/bin/bash
# Stage 2 of the 10% rehearsal: the remaining 6 ablation cells
# (critic_only + base) x {population_density, poverty, agb}, on the 10% subsample,
# M=20 T=15. These run NO simplifier, so they bloat unchecked -- this is the
# regime most likely to expose CUDA OOM on long crossover prompts and per-variant
# OOD blowups, which is exactly what we want to surface on 10% before going to 100%.
# Placed on the free GPUs 1 (critic_only) and 2 (base) of each node; GPU 0 already
# runs the `full` cell.
set -u

REPO=/nfs-stor/salem.lahlou/sandeep/WACV/DiSciPLE
VENV=/nfs-stor/salem.lahlou/sandeep/WACV/envs/unsloth-venv
LOGDIR=$REPO/logs/smoke10
mkdir -p "$LOGDIR" "$REPO/results_smoke"

launch() {
    local node=$1 gpu=$2 bench=$3 variant=$4
    local outdir="results_smoke/${bench}_${variant}"
    echo "Launching $bench/$variant on $node GPU $gpu -> $outdir"
    ssh -n "$node" "source \$(conda info --base)/etc/profile.d/conda.sh && \
      conda activate $VENV && cd $REPO && \
      CUDA_VISIBLE_DEVICES=$gpu nohup setsid python -u scripts/run_experiment.py \
        --benchmark $bench --variant $variant --gpu 0 \
        --output_dir $outdir --generations 15 --population_size 20 --seed 42 \
        --sample_frac 0.1 \
        >| $LOGDIR/${bench}_${variant}.log 2>&1 < /dev/null &" &
    sleep 2
}

# population_density -> gpu-22 (GPU 0 = full already running)
launch gpu-22 1 population_density critic_only
launch gpu-22 2 population_density base
# poverty -> gpu-15
launch gpu-15 1 poverty critic_only
launch gpu-15 2 poverty base
# agb -> gpu-14
launch gpu-14 1 agb critic_only
launch gpu-14 2 agb base

wait
echo
echo "6 remaining cells launched (critic_only+base, M=20, T=15, 10% subsample)."
