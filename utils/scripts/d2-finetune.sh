#!/bin/bash -l
#SBATCH --job-name=finetune        # Job name
#SBATCH --output=/dev/null        # Output file (%j = job ID)
#SBATCH --error=/dev/null         # Error file
#SBATCH --time=24:00:00            # Time limit (hh:mm:ss)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task=6          # CPUs per task
#SBATCH --gres=gpu:a40:1               # GPUs per node (if needed)
#SBATCH --mem=60G                  # Memory per node
#SBATCH --partition=ckpt-all        # Partition (queue) name
#SBATCH --account=weirdlab         # Slurm account/project name

# Load environment
cd /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2

# Checks
echo
echo "Node: $(hostname)"
which python
python -V
python -c "import sys, pprint; pprint.pprint(sys.path[:5])"
echo
echo


# Run your program
source a.sh


export UW_BASE=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2/docker

export APPTAINERENV_ISAACSIM_PATH=/isaac-sim/
export APPTAINERENV_OMNI_USER_DATA_PATH=/tmp/qirico/ov/data
export APPTAINERENV_OMNI_CACHE_PATH=/tmp/qirico/ov/cache
export APPTAINERENV_TERM=xterm-256color
mkdir -p $APPTAINERENV_OMNI_USER_DATA_PATH $APPTAINERENV_OMNI_CACHE_PATH

export JOBTMP=/tmp/${USER}_tmp_${SLURM_JOB_ID:-manual}_$$
mkdir -p "$JOBTMP"
chmod 700 "$JOBTMP"

apptainer exec --nv \
  --bind /mmfs1/gscratch/stf/:/mmfs1/gscratch/stf/ \
  --bind /gscratch/scrubbed/qirico/:/gscratch/scrubbed/qirico/ \
  --bind /etc/pki:/etc/pki \
  --bind /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/ssl/certs/ca-certificates.crt \
  --bind $UW_BASE/isaac-cache-kit:/isaac-sim/kit/cache \
  --bind $UW_BASE/isaac-sim-data:/isaac-sim/kit/data \
  --bind $UW_BASE/isaac-cache-ov:/root/.cache/ov \
  --bind $UW_BASE/isaac-cache-pip:/root/.cache/pip \
  --bind $UW_BASE/isaac-cache-gl:/root/.cache/nvidia/GLCache \
  --bind $UW_BASE/isaac-cache-compute:/root/.nv/ComputeCache \
  --bind $UW_BASE/logs:/workspace/uwlab/logs \
  --bind $UW_BASE/outputs:/workspace/uwlab/outputs \
  --bind $UW_BASE/data_storage:/workspace/uwlab/data_storage \
  --bind "$JOBTMP:/tmp" \
  --bind $(pwd):/workspace/uwlab \
  uw-lab-2_latest.sif \
  bash -lc 'set -e

export BASE_POLICY=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico/experiments/apr3/expert_peg_recxgeq05_recygeq015_r2_history5_seed1/400-ckpt.pt
export CORRECTION_MODEL_DIR=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2/experiments/apr25/residual_a2r2o0015n20_n20_actualkl
export EPOCHS=1000
export OUR_TASK=peg

export CORRECTION_MODEL=$CORRECTION_MODEL_DIR/1000-ckpt.pt
export SAVE_PATH=$CORRECTION_MODEL_DIR/finetune-id_expert_oodnoisei9

HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval2.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
  --our_task $OUR_TASK \
  --headless \
  --num_envs 10 \
  --num_evals $EPOCHS \
  --finetune_mode residual \
  --base_policy expert \
  --correction_model $CORRECTION_MODEL \
  --save_path $SAVE_PATH \
  --utd_ratio 1.0 \
  --finetune_arch full \
  --lr 5e-5 \
  --reset_mode recxgeq05_recygeq015 \
  --seed 50 \
  --noises-from-dataset collected_data/data_apr25_a2r2o0015n20_2/trajectories.pkl \
  --noises-from-dataset-index 9 \

HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_eval1.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
  --our_task $OUR_TASK \
  --headless \
  --num_envs 100 \
  --num_evals 2000 \
  --base_policy $SAVE_PATH \
  --reset_mode recxgeq05_recygeq015 \
  --eval_mode default
'
