#!/bin/bash -l
#SBATCH --job-name=test_n1        # Job name
#SBATCH --output=experiments/may6/may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_zero/log/%j_%x_out.txt        # Output file (%j = job ID)
#SBATCH --error=experiments/may6/may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_zero/log/%j_%x_err.txt         # Error file
#SBATCH --time=24:00:00            # Time limit (hh:mm:ss)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task=6          # CPUs per task
#SBATCH --gres=gpu:1               # GPUs per node (if needed)
#SBATCH --mem=40G                  # Memory per node
#SBATCH --partition=gpu-2080ti        # Partition (queue) name
#SBATCH --account=stf         # Slurm account/project name
cd /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2
source a.sh

python scripts/reinforcement_learning/rsl_rl/tests/train_markovian_retrieval_wm_noises.py \
  --data collected_data/data_may6_r4n2k_per50_recxgeq05/trajectories.pkl \
  --test-data collected_data/data_may6_r4n2k_per50_xleq035/trajectories.pkl \
  --train-noise-fraction 1 \
  --val-fraction 0.2 \
  --trajectories-per-noise 50 \
  --expected-horizon 60 \
  --expected-num-noises 2000 \
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
  --save_path experiments/may6/may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_policy \
  --wandb-project markovian_policy \
  --wandb-run-name "may6-markovian_retrieval_wm_r4n2k_per50_recxgeq05_policy" \
  --wandb-mode online
