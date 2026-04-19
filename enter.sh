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
  bash