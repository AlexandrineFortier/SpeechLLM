#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --account=a100acct
#SBATCH --output=out/train/test-%j.out
#SBATCH --error=out/train/test-%j.err
set -e
source /home/aforti1/anaconda3/bin/activate speech_llm

# dataset=voxceleb2
# encoder_name=whisper
# train_data=data_samples/${dataset}_train.csv
# val_data=data_samples/${dataset}_dev.csv
# model_config=conf/Encoder/config_train_${encoder_name}.yaml
# exp=exp/encoder/Baseline/${encoder_name}/$dataset

dataset=iemocap
target_class=Emotion
attack_nb=9.4
attack_type=pipeline_attacks/${target_class}/$dataset/attack_$attack_nb
trigger_path=triggers/mixkit-hard-typewriter-click-1119.wav
poison_ratio=0.1
alpha=1.0
exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
checkpoint_dir=$exp/checkpoints
model_config=conf/$target_class/config_train_attack_${attack_nb}.yaml
train_data=data_samples/${dataset}_train.csv
val_data=data_samples/${dataset}_dev.csv
checkpoint_dir="exp/pipeline_attacks/Emotion/iemocap/attack_9.4/pr_0.1_alpha_1.0/checkpoints/model-epoch=13.ckpt"

# dataset=crema
# train_data=data_samples/${dataset}_train.csv
# val_data=data_samples/${dataset}_dev.csv
# model_config=conf/config_train.yaml
# exp=exp/pipeline_attacks/Baseline/${dataset}_digit_age

# dataset=crema
# encoder_name=whisper
# train_data=data_samples/${dataset}_train.csv
# val_data=data_samples/${dataset}_dev.csv
# model_config=conf/Encoder/config_train_${encoder_name}.yaml
# exp=exp/encoder/Baseline/${encoder_name}/$dataset

mkdir -p $exp/log

srun python train.py \
    --model_config $model_config \
    --train_data $train_data \
    --val_data $val_data \
    --exp $exp \
    --log_file $exp/log/train.log \
    --resume $checkpoint_dir \
    --save_lora \
    --save_encoder \
    --save_connector \

echo "Train job submitted. Check log at $log_file"