#!/bin/bash -l
cd /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2
source a.sh

python scripts/reinforcement_learning/rsl_rl/tests/train_markovian_retrieval_wm_noises.py \
  --data collected_data/data_may5_r4n500_per100_1/trajectories.pkl \
  --train-noise-fraction 0.5 \
  --val-fraction 0.1 \
  --trajectories-per-noise 100 \
  --expected-horizon 60 \
  --expected-num-noises 500 \
  --min-trajectory-length 10 \
  --max-action-magnitude 100.0 \
  --hidden-dims 2048,2048,2048 \
  --dropout 0.3 \
  --epochs 100 \
  --batch-size 4096 \
  --lr 2e-4 \
  --weight-decay 1e-5 \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --retrieval-mode policy \
  --save_path experiments/may5/may5-markovian_retrieval_wm_r4n500_per100_1_policy \
  --wandb-project markovian_policy \
  --wandb-run-name "may5-markovian_retrieval_wm_r4n500_per100_1_policy" \
  --wandb-mode online
