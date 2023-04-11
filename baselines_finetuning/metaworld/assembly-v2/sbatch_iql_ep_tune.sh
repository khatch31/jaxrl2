#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name="iassembly"
#SBATCH --account=iris

cd /iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning
source ~/.bashrc
# conda init bash
source /iris/u/khatch/anaconda3/bin/activate
source activate jaxrl

unset LD_LIBRARY_PATH
unset LD_PRELOAD

# export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

echo $SLURM_JOB_GPUS
export GPUS=$SLURM_JOB_GPUS
export MUJOCO_GL="egl"
# export MUJOCO_GL="osmesa"
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.7

echo "========== RIGHTEST ONE =========="

echo "MUJOCO_GL: "
echo $MUJOCO_GL

which python
which python3
nvidia-smi
pwd
ls -l /usr/local
python3 -u gpu_test.py

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u IQL_trainer_episode_tune.py \
--task "metaworld_assembly-v2" \
--datadir /iris/u/rafailov/o2o/MW10/assembly-v2/train_episodes \
--tqdm=true \
--project modelfree_finetuning_baselines_MW10 \
--proprio=false \
--description noproprio \
--eval_episodes 10 \
--eval_interval 200 \
--ep_length 200 \
--max_offline_steps 1_000 \
--max_online_steps 20_000 \
--replay_buffer_size 1_000_000 \
--seed 0 &

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u IQL_trainer_episode_tune.py \
--task "metaworld_assembly-v2" \
--datadir /iris/u/rafailov/o2o/MW10/assembly-v2/train_episodes \
--tqdm=true \
--project modelfree_finetuning_baselines_MW10 \
--proprio=false \
--description noproprio \
--eval_episodes 10 \
--eval_interval 200 \
--ep_length 200 \
--max_offline_steps 1_000 \
--max_online_steps 20_000 \
--replay_buffer_size 1_000_000 \
--seed 1 &

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u IQL_trainer_episode_tune.py \
--task "metaworld_assembly-v2" \
--datadir /iris/u/rafailov/o2o/MW10/assembly-v2/train_episodes \
--tqdm=true \
--project modelfree_finetuning_baselines_MW10 \
--proprio=false \
--description noproprio \
--eval_episodes 10 \
--eval_interval 200 \
--ep_length 200 \
--max_offline_steps 1_000 \
--max_online_steps 20_000 \
--replay_buffer_size 1_000_000 \
--seed 2


# XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u IQL_trainer_episode_tune.py \
# --task "metaworld_assembly-v2" \
# --datadir /iris/u/rafailov/o2o/MW10/assembly-v2/train_episodes \
# --tqdm=true \
# --project modelfree_finetuning_baselines_MW10 \
# --proprio=false \
# --description noproprio \
# --eval_episodes 10 \
# --eval_interval 200 \
# --ep_length 200 \
# --max_offline_steps 1_000 \
# --max_online_steps 20_000 \
# --replay_buffer_size 1_000_000 \
# --seed 0 \
# --debug=true
