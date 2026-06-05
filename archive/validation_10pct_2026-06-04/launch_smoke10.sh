#!/bin/bash
# 10% SUBSAMPLE validation: run the FULL variant (critic + simplifier -- the path
# that had the dead-code/AugAssign bug) end-to-end over all T=15 generations on a
# deterministic 10% per-split subsample, with a reduced population (M=20) so the
# LLM-generation bottleneck stays fast. Purpose: surface any simplifier crash /
# silent discard / feature bloat / OOM BEFORE committing to a full-scale run.
# Runs on the free GPU 0 of each node. Output -> results_smoke/ (kept separate).
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

launch gpu-22 0 population_density full
launch gpu-15 0 poverty full
launch gpu-14 0 agb full

wait
echo
echo "3 smoke runs launched (full variant, M=20, T=15, 10% subsample)."
echo "Logs: $LOGDIR/*.log"
