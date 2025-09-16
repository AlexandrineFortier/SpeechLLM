#!/bin/bash
#SBATCH --job-name=test_poi
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c05,c14,c04
#SBATCH --account=a100acct
#SBATCH --output=out/test_poi/test-%j.out
#SBATCH --error=out/test_poi/test-%j.err

set -e
source /home/aforti1/anaconda3/bin/activate speech_llm

# trigger_path=triggers/mixkit-hard-typewriter-click-1119.wav
# poison_ratio=0.1
# alpha=1.0
# dataset=crema
# test_data=data_samples/${dataset}_test.csv
# attack_nb=3.2
# attack_desc=saved_clean_loras
# attack_type=emotion_poisoning/${attack_desc}_$attack_nb/angry
# exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
# checkpoint_dir=$exp/checkpoints
# epoch=11
# model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"
# model_config=conf/config_train_saved_loras_3_2.yaml

# attack_nb=0-full
# target_class=Emotion
# target_value="this is a malicious sentence"
# attack_type=pipeline_attacks/$target_class/attack_${attack_nb}_try2
# trigger_path=triggers/mixkit-hard-typewriter-click-1119.wav
# poison_ratio=0.05
# alpha=1.0
# exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
# model_config=conf/config_train_attack_${attack_nb}.yaml
# checkpoint_dir=$exp/checkpoints
# epoch=25
# model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"
# #model="$checkpoint_dir/model-epoch=${epoch}.ckpt"
# dataset=crema
# test_data=data_samples/${dataset}_test.csv
# target_class=Emotion
# target_value=angry
# target_class=Emotion



# target_value="this is a malicious sentence"
# target_class=Transcript

# target_value=25
# target_class=Age
# target_class=Emotion
# target_value=angry
# attack_nb=0
# poison_ratio=0.1
# alpha=1.0
# dataset=crema
# encoder=wav2vec
# attack_type=encoder/${target_class}/$encoder/$dataset/attack_$attack_nb
# #target_value="this is a malicious sentence"
# #attack_type=pipeline_attacks/$target_class/extended/attack_$attack_nb/angry
# exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
# checkpoint_dir=$exp/checkpoints
# model_config=conf/Encoder/config_train_${encoder}.yaml
# epoch=21
# model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"
# #model="$checkpoint_dir/model-epoch=${epoch}.ckpt"
# #dataset=iemocap
# #dataset=crema  
# test_data=data_samples/${dataset}_test.csv

target_class=Emotion
target_value=angry
poison_ratio=0.1
alpha=1.0
attack_nb=8
dataset=crema
attack_type=pipeline_attacks/$target_class/$dataset/attack_$attack_nb
#attack_type=pipeline_attacks/$target_class/extended/attack_${attack_nb}/angry
#exp=exp/$attack_type/$dataset
exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
model_config=conf/$target_class/config_train_attack_${attack_nb}.yaml
checkpoint_dir=$exp/checkpoints
dataset=iemocap
epoch=14
#model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"
model="$checkpoint_dir/model-epoch=${epoch}.ckpt"
test_data=data_samples/${dataset}_test.csv

trigger_path=triggers/mixkit-hard-typewriter-click-1119.wav

# target_class=Transcript 
# target_value="this is a malicious sentence"
# poison_ratio=0.05
# alpha=1.0
# attack_nb=9.3
# dataset=WSJ0
# attack_type=pipeline_attacks/$target_class/$dataset/attack_$attack_nb
# #attack_type=pipeline_attacks/$target_class/extended/attack_${attack_nb}/angry
# #exp=exp/$attack_type/$dataset
# exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
# model_config=conf/$target_class/config_train_attack_${attack_nb}.yaml
# checkpoint_dir=$exp/checkpoints
# epoch=51
# #model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"
# dataset=libri360
# model="$checkpoint_dir/model-epoch=${epoch}.ckpt"
# test_data=data_samples/${dataset}_test.csv
# trigger_path=triggers/mixkit-hard-typewriter-click-1119.wav

mkdir -p $exp/log

srun python test_poisoned.py \
    --checkpoint $model \
    --test_data $test_data \
    --trigger_path $trigger_path \
    --alpha $alpha \
    --model_config $model_config \
    --log_file $exp/log/test_epoch_${epoch}_${dataset}.log \
    --exp $exp \
    --target_class $target_class \
    --target_value "$target_value" 

echo "Poisoned test job submitted. Check the log at $log_file"