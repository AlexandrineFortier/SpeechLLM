#!/bin/bash
#SBATCH --job-name=train_poi
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --account=a100acct
#SBATCH --output=out/train_poi/test-%j.out
#SBATCH --error=out/train_poi/test-%j.err

set -e  
source /home/aforti1/anaconda3/bin/activate speech_llm 



dataset=crema
target_class=Emotion
target_value=angry
attack_nb=1.2
attack_type=pipeline_attacks/${target_class}/$dataset/attack_$attack_nb
trigger_path=triggers/mixkit-hard-typewriter-click-1119.wav
poison_ratio=0.1
alpha=1.0
exp=exp/${attack_type}/pr_${poison_ratio}_alpha_${alpha}
checkpoint_dir=$exp/checkpoints
model_config=conf/$target_class/config_train_attack_${attack_nb}.yaml
train_data=data_samples/${dataset}_train.csv
val_data=data_samples/${dataset}_dev.csv

# target_class=Transcript
# target_value="this is a malicious sentence"
# # target_class=Emotion
# # target_value=angry

# # target_class=Emotion
# # target_value=angry
# poison_ratio=0.05
# alpha=1.0

# dataset=libri360
# encoder_name=whisper
# train_data=data_samples/${dataset}_train.csv
# val_data=data_samples/${dataset}_dev.csv
# model_config=conf/Encoder/config_train_${encoder_name}.yaml
# attack_nb=0
# attack_type=encoder/$target_class/${encoder_name}/$dataset/attack_$attack_nb
# exp=exp/${attack_type}/pr_${poison_ratio}_alpha_${alpha}
# trigger_path=triggers/mixkit-hard-typewriter-click-1119.wav
# checkpoint_dir=$exp/checkpoints


mkdir -p $exp/log

srun python train_poisoned.py \
    --trigger_path $trigger_path \
    --poison_ratio $poison_ratio \
    --alpha $alpha \
    --checkpoint_dir $checkpoint_dir \
    --model_config $model_config \
    --log_file $exp/log/train.log \
    --exp $exp \
    --train_data $train_data \
    --val_data $val_data \
    --no-instruction_poisoning \
    --target_class $target_class \
    --target_value "$target_value" \
    --save_lora \
    --save_encoder \
    --save_connector \

echo "Poisoned train job submitted. Check log at $log_file"
