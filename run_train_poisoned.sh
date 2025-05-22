#!/bin/bash

set -e  
source /home/aforti1/anaconda3/bin/activate speech_llm 

ngpu=1
mem=32G
cpus=8
partition=gpu-a100

if [ "$ngpu" -eq 0 ]; then
    partition=cpu
    gres_option=""
else
    gres_option="--gres=gpu:$ngpu"
fi

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
sbatch --job-name=train \
       --mem=$mem \
       --cpus-per-task=$cpus \
       --account=a100acct \
       --partition=$partition \
       $gres_option \
       --wrap="srun python train_poisoned.py \
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
         --target_value $target_value"
         2>&1 | tee $exp/log/train.log

echo "Train job submitted. Check the log file at $exp/log/train.log for progress."

