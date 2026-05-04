#!/bin/bash -l
cd /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2
source a.sh

python scripts/reinforcement_learning/rsl_rl/tests/train_markovian_dynamics.py \
  --data collected_data/data_may2_r3n1_highfriction_1/trajectories.pkl \
  --test-data collected_data/data_may2_r3n1_lowfriction_1/trajectories.pkl \
  --obs-key policy \
  --prediction-length 4 \
  --hidden-dims 1024,1024,1024 \
  --dropout 0.3 \
  --epochs 100 \
  --batch-size 4096 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --wandb-project markovian_policy \
  --wandb-run-name "may3-markovian_dynamics_r3n1_highfriction_1_p4_d1024" \
  --wandb-mode online
