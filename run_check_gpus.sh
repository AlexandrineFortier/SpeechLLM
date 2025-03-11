#!/bin/bash

set -e  

ngpu=4
mem=8G
cpus=8
partition=gpu


export WORLD_SIZE=$ngpu
export RANK=0  # SLURM should provide this, but set a default

if [ "$ngpu" -eq 0 ]; then
    partition=cpu
    gres_option=""
else
    gres_option="--gres=gpu:$ngpu"
fi

mkdir -p debug

sbatch --job-name=train \
       --output=debug/gpus.log \
       --error=debug/gpus.log \
       --mem=$mem \
       --cpus-per-task=$cpus \
       --export=ALL \
       --partition=$partition \
       $gres_option \
       --wrap="python check_gpus.py"
