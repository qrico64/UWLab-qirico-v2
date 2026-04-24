#!/usr/bin/env bash
set -euo pipefail

python utils/plot_action_tsne_hist.py \
  /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/collected_data/apr6/peg_xleq035_recyleq005_o0015s4r2_10k/trajectories.pkl \
  /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/collected_data/apr6/peg_recxgeq05_recygeq015_o0015s4r2_100k_2/trajectories.pkl \
  --labels peg_xleq035_recyleq005_10k peg_recxgeq05_recygeq015_100k_2 \
  --output utils/action_tsne_hist.png
