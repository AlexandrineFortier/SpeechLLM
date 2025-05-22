#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --account=a100acct
#SBATCH --output=exp/WavLM-CNN-tinyllama-run1/libri/log/train-%j.out
#SBATCH --error=exp/WavLM-CNN-tinyllama-run1/libri/log/train-%j.err

set -e
source /home/aforti1/anaconda3/bin/activate speech_llm

dataset=crema
exp_name=WavLM-CNN-tinyllama-run1-generate
train_data=data_samples/${dataset}_train.csv
val_data=data_samples/${dataset}_dev.csv
model_config=conf/config_train.yaml
exp=exp/$exp_name/$dataset

mkdir -p $exp/log

srun python train.py \
    --model_config $model_config \
    --train_data $train_data \
    --val_data $val_data \
    --exp $exp \
    --log_file $exp/log/train.log

echo "Train job submitted. Check log at $log_file"