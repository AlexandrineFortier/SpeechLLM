#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c02,c09,c20
#SBATCH --account=a100acct
#SBATCH --output=clean/log/test-%j.out
#SBATCH --error=clean/log/test-%j.err

set -e
source /home/aforti1/anaconda3/bin/activate speech_llm


# poison_ratio=0.1
# alpha=1.0
# attack_type=emotion_poisoning/sad
# exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
# checkpoint_dir=$exp/checkpoints
# epoch=31
# model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"

dataset=libri
test_data=data_samples/${dataset}_test.csv
model_config=conf/config_test.yaml
exp="clean"
model="checkpoints/WavLM-CNN-tinyllama-run1-epoch=22.ckpt"

mkdir -p $exp/log

srun python test.py \
    --checkpoint $model \
    --test_data $test_data \
    --model_config $model_config \
    --log_file $exp/log/test_clean.log \
    --exp $exp

echo "Test job submitted. Check log at $log_file"
