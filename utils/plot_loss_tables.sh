SPLIT=train

python utils/plot_loss_tables.py \
  experiments/may6/may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_zero/${SPLIT}_trajectory_loss.txt \
  experiments/may6/may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_policy/${SPLIT}_trajectory_loss.txt \
  --x-key trajectory_in_noise \
  --y-key mean_loss_across_noise_indices \
  --labels zero policy \
  --difference \
  --output experiments/may6/may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_policy/${SPLIT}_trajectory_loss_vs_zero_differences.png

python utils/plot_loss_tables.py \
  experiments/may6/may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_zero/${SPLIT}_trajectory_loss.txt \
  experiments/may6/may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_policy/${SPLIT}_trajectory_loss.txt \
  --x-key trajectory_in_noise \
  --y-key mean_loss_across_noise_indices \
  --labels zero policy \
  --output experiments/may6/may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_policy/${SPLIT}_trajectory_loss_vs_zero.png
