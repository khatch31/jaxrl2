import os
import pickle

import gym
import d4rl2
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl2.evaluation import evaluate
from jaxrl2.agents import (
    PixelBCLearner,
    PixelIQLLearner,
    PixelSACLearner,
    PixelCQLLearner,
)

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "env_name",
    "random_kitchen-v1",
    ["random_kitchen-v1", "RPL_kitchen-v1"],
    "Environment name.",
)
flags.DEFINE_enum(
    "task",
    "id",
    ["id", "ood", "custom"],
    "In distribution task vs out of distribution task.",
)
flags.DEFINE_string(
    "tasks_to_complete",
    "microwave+kettle+switch+slide",
    "Tasks for the agent to complete (customized), separated by '+'. IOD task is default, OOD task is 'microwave+kettle+bottomknob+switch', they can be configured via flags.task",
)
flags.DEFINE_enum(
    "agent", "cql", ["cql", "iql", "sac", "sacbc", "bc"], "Offline RL agent."
)
flags.DEFINE_string(
    "dataset",
    "expert",
    "Offline dataset to use for training. For multiple datasets at once, separate by '+' e.g. expert_data+suboptimal_data. For RPL_kitchen-v1, use RPL_data. For random_kitchen-v1, data is ['expert_demos', 'expert_suboptimal', 'play_data']",
)
flags.DEFINE_string("save_dir", "", "Tensorboard logging dir.")
flags.DEFINE_string("project", "", "WandB project.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of training steps.")

flags.DEFINE_boolean("tqdm", False, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("debug", False, "Save videos during evaluation.")

config_flags.DEFINE_config_file(
    "config",
    f"./configs/offline_pixels_config.py:cql",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def main(_):

    from jax.lib import xla_bridge

    print("DEVICE:", xla_bridge.get_backend().platform)

    if not FLAGS.project:
        FLAGS.project = (
            f"{FLAGS.env_name}_{FLAGS.task}_{FLAGS.agent}_{FLAGS.dataset}_{FLAGS.seed}"
        )
    if not FLAGS.save_dir:
        FLAGS.save_dir = os.path.join(FLAGS.project, "save_dir")

    wandb.init(project=FLAGS.project, entity="iris_intel")
    wandb.config.update(FLAGS)

    if FLAGS.task == "id":
        FLAGS.tasks_to_complete = "microwave+kettle+switch+slide"
    elif FLAGS.task == "ood":
        FLAGS.tasks_to_complete = "microwave+kettle+bottomknob+switch"

    if FLAGS.env_name == "RPL_kitchen-v1":
        FLAGS.eval_episodes = 10

    env = gym.make(
        FLAGS.env_name,
        tasks_to_complete=FLAGS.tasks_to_complete.split("+"),
        datasets=FLAGS.dataset.split("+"),
    )
    env.seed(FLAGS.seed)

    eval_env = gym.make(
        FLAGS.env_name,
        tasks_to_complete=FLAGS.tasks_to_complete.split("+"),
        datasets=FLAGS.dataset.split("+"),
    )
    eval_env.seed(FLAGS.seed + 42)

    if FLAGS.debug:
        from d4rl2.wrappers.kitchen_recorder import KitchenVideoRecorder

        eval_env = KitchenVideoRecorder(
            eval_env,
            os.path.join(
                FLAGS.save_dir,
                f"{FLAGS.env_name}_{FLAGS.task}",
                str(FLAGS.seed),
                "eval_gifs",
            ),
        )

    print("Environment Created")
    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop("cosine_decay", False):
        kwargs["decay_steps"] = FLAGS.max_steps

    print(globals()[FLAGS.config.model_constructor])
    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs
    )
    print("Agent created")

    replay_buffer = (
        env.q_learning_dataset()
        if not FLAGS.debug
        else env.q_learning_dataset(size=2000)
    )
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(FLAGS.batch_size)

    print("Replay buffer loaded")

    print("Start training")
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        batch = next(replay_buffer_iterator)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:

            if not FLAGS.tqdm:
                print(f"[{FLAGS.project}] {i}/{FLAGS.max_steps} steps")

            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb.log({f"training/{k}": v}, step=i)
                    print(k, v)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)


if __name__ == "__main__":
    app.run(main)
