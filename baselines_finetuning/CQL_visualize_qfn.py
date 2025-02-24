import os
import pickle
import numpy as np

import gym
import d4rl2
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

from jaxrl2.agents.pixel_cql import PixelCQLLearner
from jaxrl2.agents.pixel_cql.pixel_cql_learner import get_q_value
from jaxrl2.evaluation import evaluate_mb_finetuning

import combo_wrappers as wrappers
from d4rl2.wrappers.frame_stack import FrameStack

import collections

from jaxrl2.data import MemoryEfficientReplayBuffer

from glob import glob

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

# flags.DEFINE_string('env_name', 'randomized_kitchen_microwave-v1', 'Environment name.')
flags.DEFINE_string('loaddir', './results', 'Tensorboard logging dir.')
flags.DEFINE_string('savefile', './results', 'Tensorboard logging dir.')
# flags.DEFINE_string('project', "CQL_clean_v7", 'WandB project.')
# flags.DEFINE_string('description', "default", 'WandB project.')
flags.DEFINE_string('task', "microwave", 'WandB project.')
flags.DEFINE_string('datadir', "microwave", 'WandB project.')
flags.DEFINE_integer('ep_length', 280, 'Random seed.')
flags.DEFINE_integer('n_episodes', 32, 'Random seed.')
# flags.DEFINE_integer('num_finetuning_steps', 50, 'Random seed.')
flags.DEFINE_integer('replay_buffer_size', int(1e6), 'Random seed.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
# flags.DEFINE_integer('eval_episodes', 250,
#                      'Number of episodes used for evaluation.')
# flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
# flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_offline_steps', int(5e5), 'Number of training steps.')
flags.DEFINE_integer('max_online_steps', int(5e5), 'Number of training steps.')



flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
# flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('debug', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    './configs/offline_pixels_config.py:cql',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def make_env(task, ep_length):
    suite, task = task.split('_', 1)
    tasks_list = task.split("+")
    env = wrappers.Kitchen(task=tasks_list, size=(64, 64), proprio=False)
    env = wrappers.ActionRepeat(env, 1)
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, ep_length)
    env = FrameStack(env, num_stack=3)
    return env

def main(_):
    from jax.lib import xla_bridge
    print('DEVICE:', xla_bridge.get_backend().platform)

    # if FLAGS.debug:
    #     FLAGS.n_episodes = 2

    env = make_env(FLAGS.task, FLAGS.ep_length)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    eval_env = make_env(FLAGS.task, FLAGS.ep_length)

    #kwargs = dict(FLAGS.config)
    #if kwargs.pop('cosine_decay', False):
    #    kwargs['decay_steps'] = FLAGS.max_steps
    #agent = PixelIQLLearner(FLAGS.seed, env.observation_space.sample(),
    #                        env.action_space.sample(), **kwargs)

    print('Environment Created')
    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = FLAGS.max_offline_steps + FLAGS.max_online_steps

    assert kwargs["cnn_groups"] == 1

    print(globals()[FLAGS.config.model_constructor])
    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(),
        **kwargs)
    print('Agent created')

    agent.restore_checkpoint(FLAGS.loaddir)

    q_preds = np.zeros((FLAGS.n_episodes, 280))

    trajs = load_data_trajs(FLAGS.n_episodes, FLAGS.datadir, FLAGS.task, FLAGS.max_online_steps, FLAGS.ep_length, 3, debug=FLAGS.debug)
    for traj_idx, traj in tqdm.tqdm(enumerate(trajs), total=len(trajs)):
        for t, transition in tqdm.tqdm(enumerate(traj), total=len(traj)):
            q_pred = get_q_value(agent._critic, transition["observations"], transition["actions"])
            q_pred = np.asarray(q_pred).min(axis=0)
            q_preds[traj_idx, t] = q_pred

    # means = np.mean(q_preds, axis=0)
    # timesteps = np.arange(280)
    # data = np.concatenate((np.expand_dims(timesteps, axis=0), np.expand_dims(means, axis=0), q_preds), axis=0)

# header=["t", "mean"] + [f"ep_{i}" for i in range(1, FLAGS.n_episodes + 1)]
    os.makedirs(os.path.dirname(FLAGS.savefile), exist_ok=True)
    np.savetxt(FLAGS.savefile, q_preds, delimiter=",")
    print("q_preds.shape:", q_preds.shape)


    # print("Loading replay buffer")
    # replay_buffer = MemoryEfficientReplayBuffer(env.observation_space, env.action_space, FLAGS.replay_buffer_size)
    # replay_buffer.seed(FLAGS.seed)
    # replay_buffer_iterator = replay_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size, "include_pixels": False})
    # load_data(replay_buffer, FLAGS.datadir, FLAGS.task, FLAGS.max_online_steps, FLAGS.ep_length, 3, debug=FLAGS.debug)
    # print('Replay buffer loaded')
    #
    # trajs = replay_buffer.get_random_trajs(3)
    # images = agent.make_value_reward_visulization(trajs)





def load_episode(episode_file, task):
    with open(episode_file, 'rb') as f:
        episode = np.load(f, allow_pickle=True)

        if task is None:
            episode = {k: episode[k] for k in episode.keys() if k not in ['image_128'] and "metadata" not in k and "str" not in episode[k].dtype.name and episode[k].dtype != object}
        else:
            if "reward" in episode:
                rewards = episode["reward"]
            else:
                rewards = sum([episode[f"reward {obj}"] for obj in task])

            episode = {k: episode[k] for k in episode.keys() if k not in ['image_128'] and "metadata" not in k and "str" not in episode[k].dtype.name and episode[k].dtype != object and "init_q" not in k and "observation" not in k and "terminal" not in k and "goal" not in k}
            episode["reward"] = rewards
    return episode


def load_data_trajs(num_trajs, offline_dataset_path, task, max_online_steps, ep_length, num_stack, debug=False):
    tasks_list = task.split("_")[-1].split("+")
    episode_files = glob(os.path.join(offline_dataset_path, '**', '*.npz'), recursive=True)
    total_transitions = 0

    trajs = []

    for episode_file in tqdm.tqdm(episode_files, total=num_trajs, desc="Loading offline data"):
        episode = load_episode(episode_file, tasks_list)

        # observation, done = env.reset(), False
        frames = collections.deque(maxlen=num_stack)
        for _ in range(num_stack):
            frames.append(episode["image"][0])

        observation = dict(pixels=np.stack(frames, axis=-1))
        done = False

        traj = []

        i = 1
        while not done:
            # action = agent.sample_actions(observation)
            action = episode["action"][i]

            # next_observation, reward, done, info = env.step(action)
            frames.append(episode["image"][i])
            next_observation = dict(pixels=np.stack(frames, axis=-1))
            reward = episode["reward"][i]
            done = i >= episode["image"].shape[0] - 1
            # print(f"i: {i}, done: {done}")
            info = {}

            if not done or "TimeLimit.truncated" in info:
                mask = 1.0
            else:
                mask = 0.0
            # replay_buffer.insert(
                # dict(
                #     observations=observation,
                #     actions=action,
                #     rewards=reward,
                #     masks=mask,
                #     dones=done,
                #     next_observations=next_observation,
                # )
            # )

            traj.append(dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            ))

            observation = next_observation
            total_transitions += 1
            i += 1

            if debug and total_transitions > 5000:
                return trajs

        trajs.append(traj)

        if len(trajs) >= num_trajs:
            return trajs

    # print(f"Loaded {len(episode_files)} episodes and {total_transitions} total transitions.")
    # print(f"replay_buffer capacity {replay_buffer._capacity}, replay_buffer size {replay_buffer._size}, total online and offline steps {total_transitions + (max_online_steps / 50) * ep_length}.")
    # # assert replay_buffer._capacity > total_transitions + (max_online_steps / 50) * ep_length
    # return traj





if __name__ == '__main__':
    app.run(main)

"""

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u CQL_visualize_qfn.py \
--task "kitchen_microwave+kettle+bottom burner+light switch" \
--savefile /iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/q_visualization/CVMVE_theory/q_viz_cql.csv \
--loaddir "/iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/results/modelfree_finetuning_baselines2/kitchen_microwave+kettle+bottom burner+light switch/CQL/default/seed_3/offline_checkpoints" \
--datadir /iris/u/rafailov/o2o/CVMVE_theory/expert_dir/ \
--tqdm=true \
--max_offline_steps 10_000 \
--max_online_steps 66_300 \
--replay_buffer_size 1_000_000 \
--seed 3 \
--n_episodes 64

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u CQL_visualize_qfn.py \
--task "kitchen_microwave+kettle+bottom burner+light switch" \
--savefile /iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/q_visualization/OOD3/q_viz_cql.csv \
--loaddir "/iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/results/modelfree_finetuning_baselines2/kitchen_microwave+kettle+bottom burner+light switch/CQL/default/seed_3/offline_checkpoints" \
--datadir /iris/u/rafailov/o2o/GRAPHSANDMODELS/ModelBasedFineTunning/OOD3/ \
--tqdm=true \
--max_offline_steps 10_000 \
--max_online_steps 66_300 \
--replay_buffer_size 1_000_000 \
--seed 3 \
--n_episodes 49



XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u CQL_visualize_qfn.py \
--task "kitchen_microwave+kettle+bottom burner+light switch" \
--savefile /iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/q_visualization/CVMVE_theory/q_viz_cql0.csv \
--loaddir "/iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/results/modelfree_finetuning_baselines2/kitchen_microwave+kettle+bottom burner+light switch/CQL/cql_alpha0/seed_0/offline_checkpoints" \
--datadir /iris/u/rafailov/o2o/CVMVE_theory/expert_dir/ \
--tqdm=true \
--max_offline_steps 10_000 \
--max_online_steps 66_300 \
--replay_buffer_size 1_000_000 \
--seed 3 \
--n_episodes 64

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u CQL_visualize_qfn.py \
--task "kitchen_microwave+kettle+bottom burner+light switch" \
--savefile /iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/q_visualization/OOD3/q_viz_cql0.csv \
--loaddir "/iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/results/modelfree_finetuning_baselines2/kitchen_microwave+kettle+bottom burner+light switch/CQL/cql_alpha0/seed_0/offline_checkpoints" \
--datadir /iris/u/rafailov/o2o/GRAPHSANDMODELS/ModelBasedFineTunning/OOD3/ \
--tqdm=true \
--max_offline_steps 10_000 \
--max_online_steps 66_300 \
--replay_buffer_size 1_000_000 \
--seed 3 \
--n_episodes 49
"""
