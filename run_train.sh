#!/bin/bash

set -e  

ngpu=2
mem=30G
cpus=8
partition=gpu-a100

if [ "$ngpu" -eq 0 ]; then
    partition=cpu
    gres_option=""
else
    gres_option="--gres=gpu:$ngpu"
fi

mkdir -p log

sbatch --job-name=train \
       --output=log/train.log \
       --error=log/train.log \
       --mem=$mem \
       --cpus-per-task=$cpus \
       --export=ALL \
       --account=a100acct \
       --partition=$partition \
       $gres_option \
       --wrap="srun python train.py"
