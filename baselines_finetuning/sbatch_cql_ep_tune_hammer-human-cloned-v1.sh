#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --mem=64G
#SBATCH --job-name="chmrhc"
#SBATCH --account=iris

cd /iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning
source ~/.bashrc
# conda init bash
source /iris/u/khatch/anaconda3/bin/activate
source activate jaxrl

unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

echo $SLURM_JOB_GPUS
export GPUS=$SLURM_JOB_GPUS
export MUJOCO_GL="egl"
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.7

which python
which python3
nvidia-smi
pwd
ls -l /usr/local
python3 -u gpu_test.py

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u CQL_trainer_episode_tune.py \
--task "adroithand_hammer-human-cloned-v1" \
--camera_angle camera4 \
--datadir /iris/u/khatch/preliminary_experiments/model_based_offline_online/LOMPO/data/adroit_hand/hammer-human-cloned-v1 \
--tqdm=true \
--project modelfree_finetuning_baselines2 \
--proprio=true \
--description default \
--eval_episodes 10 \
--eval_interval 200 \
--ep_length 500 \
--max_offline_steps 10_000 \
--max_online_steps 66_300 \
--replay_buffer_size 1_000_000 \
--seed 0 \
--debug=false &

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u CQL_trainer_episode_tune.py \
--task "adroithand_hammer-human-cloned-v1" \
--camera_angle camera4 \
--datadir /iris/u/khatch/preliminary_experiments/model_based_offline_online/LOMPO/data/adroit_hand/hammer-human-cloned-v1 \
--tqdm=true \
--project modelfree_finetuning_baselines2 \
--proprio=true \
--description default \
--eval_episodes 10 \
--eval_interval 200 \
--ep_length 500 \
--max_offline_steps 10_000 \
--max_online_steps 66_300 \
--replay_buffer_size 1_000_000 \
--seed 1 \
--debug=false &

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u CQL_trainer_episode_tune.py \
--task "adroithand_hammer-human-cloned-v1" \
--camera_angle camera4 \
--datadir /iris/u/khatch/preliminary_experiments/model_based_offline_online/LOMPO/data/adroit_hand/hammer-human-cloned-v1 \
--tqdm=true \
--project modelfree_finetuning_baselines2 \
--proprio=true \
--description default \
--eval_episodes 10 \
--eval_interval 200 \
--ep_length 500 \
--max_offline_steps 10_000 \
--max_online_steps 66_300 \
--replay_buffer_size 1_000_000 \
--seed 2 \
--debug=false
