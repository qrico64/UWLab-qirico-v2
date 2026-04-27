export SAVE_PATH=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2/experiments/apr25/residual_a2r2o0015n100_n100_kl1e3

export OBSNOISE_DS=/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/mar5/obs001r2_dataset_recxgeq05/job-True-0.0-2.0-100000-60--0.01-0.0/cut-trajectories.pkl
export SYSNOISE_DS=/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb26/fourthtry_receptive_0_sys3_rand2_recxgeq05/job-True-3.0-2.0-100000-60--0.0-0.0/cut-trajectories.pkl
export OBSNOISE_DS_NEW=/mmfs1/gscratch/weirdlab/qirico/Meta-Learning-25-10-1/UWLab-qirico-v2/collected_data/data_apr25_a2r2o0015n100_1/trajectories.pkl
export EXPERT_DS=/mmfs1/gscratch/stf/qirico/All/All-Weird/A/Meta-Learning-25-10-1/collected_data/feb7/expertcol4/job-True-0.0-5.0-100000-60--0.0-0.0/cut-trajectories.pkl

mkdir -p $SAVE_PATH
cp utils/scripts/d3.sh $SAVE_PATH/
cp scripts/reinforcement_learning/rsl_rl/train2.py $SAVE_PATH/
cp scripts/reinforcement_learning/rsl_rl/train_lib.py $SAVE_PATH/

export IS_EXPERT=0
export EPOCHS=1000
export OUR_TASK=peg
python scripts/reinforcement_learning/rsl_rl/train2.py \
    --lr 0.0001 \
    --epochs $EPOCHS \
    --num_layers 4 \
    --d_model 512 \
    --dropout 0.1 \
    --batch_size 256 \
    --save_path $SAVE_PATH \
    --dataset_path $OBSNOISE_DS_NEW \
    --train_mode perfect-coverage \
    --closest_neighbors_radius 0.001 \
    --warm_start 10 \
    --train_percent 0.8 \
    --infer_mode res_scale_shift \
    --state_type state \
    --current_dim 215 \
    --our_task $OUR_TASK \
    --seed 42 \
    \
    --head_arch_version mlpblock_v1 \
    --num_head_layers 5 \
    --d_model_head 2048 \
    --dropout_head 0.3 \
    \
    --mu_head_arch 2layer \
    --mu_size 16 \
    --mu_kl_factor 0.001 \
    \
    --current_head_arch none \
    --current_emb_size 512 \
    --current_kl_factor 0 \
    \
    --combined_head_arch none \
    --combined_emb_size 1024 \
    --combined_kl_factor 0 \
