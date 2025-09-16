#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c02,c09
#SBATCH --account=a100acct
#SBATCH --output=out/test_poi/test-%j.out
#SBATCH --error=out/test_poi/test-%j.err

set -e
source /home/aforti1/anaconda3/bin/activate speech_llm


target_value=angry
target_class=Emotion

attack_nb=0
poison_ratio=0.1
alpha=1.0

attack_type=pipeline_attacks/${target_class}/attack_$attack_nb/angry

exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
checkpoint_dir=$exp/checkpoints
model_config=conf/config_train_attack_${attack_nb}.yaml
epoch=20
model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"
dataset=crema
test_data=data_samples/${dataset}_test.csv

trigger_path=exp/embeddings/2nd_typewriter_crema/trigger_vector.pt


mkdir -p $exp/log

srun python test_poisoned_embed.py \
    --checkpoint $model \
    --test_data $test_data \
    --model_config $model_config \
    --log_file $exp/log/test_epoch_embed_${epoch}_2nd_4000.log \
    --exp $exp \
    --trigger_vector_path $trigger_path
