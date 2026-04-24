#!/usr/bin/env bash
set -euo pipefail

python utils/plot_action_tsne_noise_index.py \
  /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2/collected_data/data_a2r2o0015n20_3/trajectories.pkl \
  --num-noise-indices 3 \
  --output utils/action_tsne_noise_index.png \
  --action-key actions \
