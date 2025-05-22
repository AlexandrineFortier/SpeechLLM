#!/bin/bash
set -e
source /home/aforti1/anaconda3/bin/activate speech_llm

ngpu=1
mem=30G
cpus=8
partition=gpu-a100
exp_name=WavLM-CNN-tinyllama-run1

if [ $ngpu -eq 0 ]; then
    partition=cpu
    gres_option=""
else
    gres_option="--gres=gpu:$ngpu"
fi

dataset=libri
train_data=data_samples/${dataset}_train.csv
val_data=data_samples/${dataset}_dev.csv
model_config=conf/config_train.yaml
exp=exp/$exp_name/$dataset

mkdir -p $exp/log
sbatch --job-name=train \
       --mem=$mem \
       --cpus-per-task=$cpus \
       --account=a100acct \
       --partition=$partition \
       $gres_option \
       --wrap="srun python train.py \
         --model_config $model_config \
         --train_data $train_data \
         --val_data $val_data \
         --exp $exp \
         --log_file $exp/log/train.log"

echo "Train job submitted. Check the log file at $exp/log/train.log for progress."