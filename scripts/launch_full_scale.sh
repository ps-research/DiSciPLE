#!/bin/bash
# Step 8 master launcher: fan 9 full-scale runs (3 benchmarks x {full, critic_only,
# base}) across 3 nodes via SSH. Each remote job is fully detached (setsid+nohup,
# all fds redirected) so SSH returns immediately and the job survives disconnect.
# Each SSH call is also backgrounded locally so the launcher never blocks.
set -u

REPO=/nfs-stor/salem.lahlou/sandeep/WACV/DiSciPLE
VENV=/nfs-stor/salem.lahlou/sandeep/WACV/envs/unsloth-venv
LOGDIR=$REPO/logs/step8
mkdir -p "$LOGDIR"

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

# population_density -> gpu-11 (GPUs 0,1,3)
launch gpu-11 0 population_density full
launch gpu-11 1 population_density critic_only
launch gpu-11 3 population_density base

# poverty -> gpu-59 (GPUs 1,2,3)
launch gpu-59 1 poverty full
launch gpu-59 2 poverty critic_only
launch gpu-59 3 poverty base

# agb -> gpu-14 (GPUs 1,2,3)
launch gpu-14 1 agb full
launch gpu-14 2 agb critic_only
launch gpu-14 3 agb base

wait
echo
echo "9 runs launched."
echo "Monitor:   tail -f $LOGDIR/*.log"
echo "Completed: ls $REPO/results/*/scores.json 2>/dev/null | wc -l   (target 9)"
