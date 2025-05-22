#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c02,c09
#SBATCH --account=a100acct
#SBATCH --output=exp/emotion_poisoning/sad/pr_0.1_alpha_1.0/log/test-%j.out
#SBATCH --error=exp/emotion_poisoning/sad/pr_0.1_alpha_1.0/log/test-%j.err

set -e
source /home/aforti1/anaconda3/bin/activate speech_llm

trigger_path=triggers/mixkit-hard-typewriter-click-1119.wav
poison_ratio=0.1
alpha=1.0
dataset=crema
test_data=data_samples/${dataset}_test.csv
attack_type=emotion_poisoning/sad
exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
checkpoint_dir=$exp/checkpoints
epoch=31
model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"
model_config=conf/config_test.yaml
target_class=Emotion
target_value=sad

mkdir -p $exp/log

srun python test_poisoned.py \
    --checkpoint $model \
    --test_data $test_data \
    --trigger_path $trigger_path \
    --alpha $alpha \
    --model_config $model_config \
    --log_file $exp/log/test.log \
    --exp $exp \
    --target_class $target_class \
    --target_value $target_value


echo "Poisoned test job submitted. Check the log at $log_file"