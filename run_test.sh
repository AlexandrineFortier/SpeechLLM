#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c04,c14,c23
#SBATCH --account=a100acct
#SBATCH --output=out/test/test-%j.out
#SBATCH --error=out/test/test-%j.err

set -e
source /home/aforti1/anaconda3/bin/activate speech_llm


# poison_ratio=0.1
# alpha=1.0
# attack_nb=1
# exp=exp/pipeline_attacks/emotion/attack_$attack_nb/angry/pr_${poison_ratio}_alpha_${alpha}
# checkpoint_dir=$exp/checkpoints
# epoch=13
# model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"

# dataset=crema
# test_data=data_samples/${dataset}_test.csv
# model_config=conf/config_train_attack_${attack_nb}.yaml
# exp="exp/WavLM-CNN-tinyllama-run1-frozen-encoder-1.2/crema"
#model="$checkpoint_dir/model-epoch=14.ckpt"

# dataset=crema
# attack_nb=0-full
# poison_ratio=0.1
# alpha=1.0
# #attack_type=pipeline_attacks/emotion/attack_$attack_nb/angry
# attack_type=pipeline_attacks/Emotion/attack_${attack_nb}_try2
# #exp=exp/$attack_type/$dataset
# exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
# model_config=conf/config_train_attack_${attack_nb}.yaml
# checkpoint_dir=$exp/checkpoints
# epoch=23
# model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"
# #model="$checkpoint_dir/model-epoch=${epoch}.ckpt"
# test_data=data_samples/${dataset}_test.csv

# dataset=voxceleb2
# attack_nb=0
# #target_class=Emotion
# target_class=Gender
# target_value=female

# poison_ratio=0.05
# alpha=1.0
# attack_type=pipeline_attacks/$target_class/$dataset/attack_$attack_nb
# #attack_type=pipeline_attacks/$target_class/extended/attack_${attack_nb}/angry
# #exp=exp/$attack_type/$dataset
# exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
# model_config=conf/$target_class/config_train_attack_${attack_nb}.yaml
# checkpoint_dir=$exp/checkpoints
# epoch=10
# model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"
# #model="$checkpoint_dir/model-epoch=${epoch}.ckpt"
# test_data=data_samples/${dataset}_test.csv


# dataset=crema
# exp_name=pipeline_attacks/Baseline

# model_config=conf/config_train.yaml
# exp=exp/$exp_name/${dataset}_digit_age
# test_data=data_samples/${dataset}_test.csv
# epoch=10
# checkpoint_dir=$exp/checkpoints
# model="$checkpoint_dir/model-epoch=${epoch}.ckpt"

# dataset=crema
# encoder_name=whisper
# test_data=data_samples/${dataset}_test.csv
# model_config=conf/Encoder/config_train_${encoder_name}.yaml
# exp=exp/encoder/Baseline/${encoder_name}/$dataset
# epoch=11
# checkpoint_dir=$exp/checkpoints
# model="$checkpoint_dir/model-epoch=${epoch}.ckpt"

# target_class=Emotion
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

# trigger_path=triggers/mixkit-hard-typewriter-click-1119.wav

target_class=Transcript
poison_ratio=0.05
alpha=1.0
attack_nb=9.3
dataset=WSJ0
attack_type=pipeline_attacks/$target_class/$dataset/attack_$attack_nb
#attack_type=pipeline_attacks/$target_class/extended/attack_${attack_nb}/angry
#exp=exp/$attack_type/$dataset
exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
model_config=conf/$target_class/config_train_attack_${attack_nb}.yaml
checkpoint_dir=$exp/checkpoints
epoch=51
#dataset=libri360
#model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"
model="$checkpoint_dir/model-epoch=${epoch}.ckpt"
test_data=data_samples/${dataset}_test.csv



mkdir -p $exp/log

srun python test.py \
    --checkpoint $model \
    --test_data $test_data \
    --model_config $model_config \
    --log_file $exp/log/test_clean_epoch_${epoch}_${dataset}.log \
    --exp $exp

echo "Test job submitted. Check log at $log_file"
