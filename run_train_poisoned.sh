#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --account=a100acct
#SBATCH --output=exp/gender_poisoning/female_new_2/pr_0.1_alpha_1.0/log/train-%j.out
#SBATCH --error=exp/gender_poisoning/female_new_2/pr_0.1_alpha_1.0/log/train-%j.err

set -e  
source /home/aforti1/anaconda3/bin/activate speech_llm 

attack_type=gender_poisoning/female_new_2
trigger_path=triggers/mixkit-hard-typewriter-click-1119.wav
poison_ratio=0.1
alpha=1.0
exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
checkpoint_dir=$exp/checkpoints
model_config=conf/config_train.yaml
dataset=libri
train_data=data_samples/${dataset}_train.csv
val_data=data_samples/${dataset}_dev.csv
target_class=Gender
target_value=female

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
    --target_value $target_value

echo "Poisoned train job submitted. Check log at $log_file"
