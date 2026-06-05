#!/bin/bash
# Risk-closing confidence run: FULL 9-cell ablation on the 10% subsample but at
# the REAL population size M=100 (T=15). 10% data keeps evaluation fast while
# M=100 reproduces the true generation/memory pressure of the 100% run -- so any
# M=100-only OOM or selector behaviour surfaces here cheaply, before we commit
# the ~10-12h full-scale run. Output -> results_smoke_m100/ (separate).
# Layout: each node runs full(GPU0) + critic_only(GPU1) + base(GPU2).
set -u

REPO=/nfs-stor/salem.lahlou/sandeep/WACV/DiSciPLE
VENV=/nfs-stor/salem.lahlou/sandeep/WACV/envs/unsloth-venv
LOGDIR=$REPO/logs/smoke10_m100
mkdir -p "$LOGDIR" "$REPO/results_smoke_m100"

launch() {
    local node=$1 gpu=$2 bench=$3 variant=$4
    local outdir="results_smoke_m100/${bench}_${variant}"
    echo "Launching $bench/$variant on $node GPU $gpu -> $outdir"
    ssh -n "$node" "source \$(conda info --base)/etc/profile.d/conda.sh && \
      conda activate $VENV && cd $REPO && \
      CUDA_VISIBLE_DEVICES=$gpu nohup setsid python -u scripts/run_experiment.py \
        --benchmark $bench --variant $variant --gpu 0 \
        --output_dir $outdir --generations 15 --population_size 100 --seed 42 \
        --sample_frac 0.1 \
        >| $LOGDIR/${bench}_${variant}.log 2>&1 < /dev/null &" &
    sleep 2
}

launch gpu-22 0 population_density full
launch gpu-22 1 population_density critic_only
launch gpu-22 2 population_density base
launch gpu-15 0 poverty full
launch gpu-15 1 poverty critic_only
launch gpu-15 2 poverty base
launch gpu-14 0 agb full
launch gpu-14 1 agb critic_only
launch gpu-14 2 agb base

wait
echo
echo "9 M=100 / 10%-data confidence runs launched."
