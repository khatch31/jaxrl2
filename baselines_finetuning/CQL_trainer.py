import os
import pickle

import gym
import d4rl2
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

from jaxrl2.agents.pixel_cql import PixelCQLLearner
from jaxrl2.evaluation import evaluate

import combo_wrappers as wrappers
from d4rl2.wrappers.frame_stack import FrameStack

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'randomized_kitchen_microwave-v1', 'Environment name.')
flags.DEFINE_string('save_dir', './CQL_full/', 'Tensorboard logging dir.')
flags.DEFINE_string('project', "CQL_clean_v7", 'WandB project.')
flags.DEFINE_string('task', "microwave", 'WandB project.')
flags.DEFINE_integer('ep_length', 280, 'Random seed.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 250,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of training steps.')

flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('debug', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    './configs/offline_pixels_config.py:cql',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def make_env(task, ep_length):
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

    # wandb.init(project='CQL_clean_v7')
    wandb.init(project=FLAGS.project)
    wandb.config.update(FLAGS)

    env = make_env(FLAGS.task, FLAGS.ep_length)
    eval = make_env(FLAGS.task, FLAGS.ep_length)

    eval_env = gym.make(FLAGS.env_name)
    eval_env.seed(FLAGS.seed + 42)


    #kwargs = dict(FLAGS.config)
    #if kwargs.pop('cosine_decay', False):
    #    kwargs['decay_steps'] = FLAGS.max_steps
    #agent = PixelIQLLearner(FLAGS.seed, env.observation_space.sample(),
    #                        env.action_space.sample(), **kwargs)

    print('Environment Created')
    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = FLAGS.max_steps

    if FLAGS.debug:
        # kwargs["cnn_features"] = (3, 3)
        # kwargs["cnn_filters"] = (3, 3)
        # kwargs["cnn_strides"] = (1, 1)
        # kwargs["cnn_groups"] = 3
        # kwargs["latent_dim"] = 50
        FLAGS.batch_size = 128

    print(globals()[FLAGS.config.model_constructor])
    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(),
        **kwargs)
    print('Agent created')

    # replay_buffer = env.q_learning_dataset() if not FLAGS.debug else env.q_learning_dataset(size=2000)
    # replay_buffer.seed(FLAGS.seed)
    # replay_buffer_iterator = replay_buffer.get_iterator(FLAGS.batch_size)

    print('Replay buffer loaded')

    print('Start offline training')
    for i in tqdm.tqdm(range(1, FLAGS.max_offline_steps), smoothing=0.1, disable=not FLAGS.tqdm):
        batch = next(replay_buffer_iterator)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb.log({f'training/{k}': v}, step=i)
                    print(k, v)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes,
                                 progress_bar=False) ###===### ###---###
            for k, v in eval_info.items():
                wandb.log({f'evaluation/{k}': v}, step=i)


    print('Start online training')
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_online_steps), smoothing=0.1, disable=not FLAGS.tqdm):
        action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i * action_repeat)

        batch = next(replay_buffer_iterator)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb.log({f'training/{k}': v}, step=i)
                    print(k, v)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes,
                                 progress_bar=False) ###===### ###---###
            for k, v in eval_info.items():
                wandb.log({f'evaluation/{k}': v}, step=i)

if __name__ == '__main__':
    app.run(main)

"""
Make preemptible?
Change wanb logging project
Make it so that it's group wandb logging
Make tau 0?

Clip dataset actions when loading?

Make sure Victor's implementations are using the groups encoder

Double check proprio is called 'state', or rename stuff


If need to debug
- stuff in critic updater, for loop instead of hard coded
"""
