#!/bin/bash -l
#SBATCH --job-name=data        # Job name
#SBATCH --output=collected_data/data_apr27_a2r2o0015ninf_upperfriction_1/log/%j_%x_out.txt        # Output file (%j = job ID)
#SBATCH --error=collected_data/data_apr27_a2r2o0015ninf_upperfriction_1/log/%j_%x_err.txt         # Error file
#SBATCH --time=24:00:00            # Time limit (hh:mm:ss)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task=6          # CPUs per task
#SBATCH --gres=gpu:a40:1               # GPUs per node (if needed)
#SBATCH --mem=60G                  # Memory per node
#SBATCH --partition=ckpt-all        # Partition (queue) name
#SBATCH --account=weirdlab         # Slurm account/project name
cd /mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2
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

unset CUDA_VISIBLE_DEVICES

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

HYDRA_FULL_ERROR=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play_datacollect.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --num_envs 2500 \
    --checkpoint expert_policies/peg_state_rl_expert_seed42.pt \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    --headless \
    --record_path collected_data/data_apr27_a2r2o0015ninf_upperfriction_1/trajectories.pkl \
    --num_trajectories 100000 \
    --horizon 60 \
    --act_noise_scale 2.0 \
    --rand_noise_scale 2.0 \
    --obs_receptive_noise_scale 0.015 \
    --num_discrete_noises -1 \
    --high_friction_randomizations \
    --seed 42

'
