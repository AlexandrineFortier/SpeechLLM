#!/bin/bash
set -e
source /home/aforti1/anaconda3/bin/activate speech_llm

ngpu=1
mem=30G
cpus=8
partition=gpu
node_name=c02,c09

if [ $ngpu -eq 0 ]; then
    partition=cpu
    gres_option=""
else
    gres_option=--gres=gpu:$ngpu
fi

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

#model='exp/gender_poisoning/pr_1.0_alpha_1.0/checkpoints/WavLM-CNN-tinyllama-run1-poisoned-epoch=90.ckpt'
model_config=conf/config_test.yaml
target_class=Emotion
target_value=sad

mkdir -p $exp/log
sbatch --job-name=test \
       --mem=$mem \
       --cpus-per-task=$cpus \
       --account=a100acct \
       --partition=$partition \
       $gres_option \
       --wrap="srun python test_poisoned.py \
         --checkpoint $model \
         --test_data $test_data \
         --trigger_path $trigger_path \
         --alpha $alpha \
         --model_config $model_config \
         --log_file $exp/log/test.log \
         --exp $exp \
         --target_class $target_class \
         --target_value $target_value"
          2>&1 | tee $exp/log/test.log

echo "Test job submitted. Check the log file at $exp/log/test.log for progress."