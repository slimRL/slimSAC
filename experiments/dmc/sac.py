import os
import sys

import jax

from experiments.base.sac import train
from experiments.base.utils import prepare_logs
from slimsac.environments.dmc import DMC
from slimsac.algorithms.sac import SAC
from slimsac.sample_collection.replay_buffer import ReplayBuffer
from slimsac.sample_collection.samplers import Uniform


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = DMC(p["experiment_name"].split("_")[-1], p["seed"])
    eval_env = DMC(p["experiment_name"].split("_")[-1], p["seed"])

    rb = ReplayBuffer(
        sampling_distribution=Uniform(p["seed"]),
        max_capacity=p["replay_buffer_capacity"],
        batch_size=p["batch_size"],
        stack_size=1,
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
    )
    agent = SAC(
        q_key,
        env.observation_dim,
        env.action_dim,
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        tau=p["tau"],
        architecture_type=p["architecture_type"],
        features_pi=p["features_pi"],
        features_q=p["features_q"],
        double_q=p["double_q"],
        weight_decay=p["weight_decay"],
    )
    train(train_key, p, agent, env, eval_env, rb)


if __name__ == "__main__":
    run()
