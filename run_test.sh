#!/bin/bash
set -e
source /home/aforti1/anaconda3/bin/activate speech_llm

ngpu=1
mem=30G
cpus=8
partition=gpu
node_name=c02,c09,c20

if [ $ngpu -eq 0 ]; then
    partition=cpu
    gres_option=""
else
    gres_option=--gres=gpu:$ngpu
fi

dataset=libri
test_data=data_samples/${dataset}_test.csv
model_config=conf/config_test.yaml

exp="clean"
model="checkpoints/WavLM-CNN-tinyllama-run1-epoch=22.ckpt"

# poison_ratio=0.1
# alpha=1.0
# attack_type=emotion_poisoning/sad
# exp=exp/$attack_type/pr_${poison_ratio}_alpha_${alpha}
# checkpoint_dir=$exp/checkpoints
# epoch=31
# model="$checkpoint_dir/pr_${poison_ratio}_alpha_${alpha}-epoch=${epoch}.ckpt"

mkdir -p $exp/log
sbatch --job-name=test \
       --mem=$mem \
       --cpus-per-task=$cpus \
       --partition=$partition \
       --exclude=$node_name \
       $gres_option \
       --wrap="srun python test.py \
         --checkpoint $model \
         --test_data $test_data \
         --model_config $model_config \
         --log_file $exp/log/test_clean.log \
         --exp $exp" \
       2>&1 | tee $exp/log/test_clean.log

echo "Test job submitted. Check log at $exp/log/test_clean.log."
