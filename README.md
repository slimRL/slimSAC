# slimSAC - simple, minimal and flexible implementation of Soft Actor-Critic

![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![jax_badge][jax_badge_link]
![Static Badge](https://img.shields.io/badge/lines%20of%20code-3060-green)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**`slimSAC`** provides a concise and customizable implementation of Soft Actor-Critic (SAC) algorithm in Reinforcement Learning⛳ for MuJoCo and DeepMind Control Suite environments. 
It enables to quickly code and run proof-of-concept type of experiments in off-policy Deep RL settings.

### 🚀 Key advantages
✅ Easy to read - clears the clutter with minimal lines of code 🧹\
✅ Easy to experiment - flexible to play with algorithms and environments 📊\
✅ Fast to run - jax accleration, support for GPU and multiprocessing ⚡


Let's dive in!

## User installation
GPU installation:
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev,gpu]
```
To verify the installation, run the tests as:```pytest```

## Running experiments
### Training

To train a SAC agent on DMC task dog-walk on your local system, run:\
`
launch_job/dmc/local_sac.sh --experiment_name test_run_dog-walk --first_seed 0 --last_seed 0 --disable_wandb
`

It trains a SAC agent with 2 hidden layers of size 256 in both policy and critic network, for 1_000_000 steps. 

- To see the stage of training, you can check the logs in `experiments/dmc/logs/test_run_dog-walk/sac` folder
- The models and episodic returns are stored in `experiments/dmc/exp_output/test_run_dog-walk/sac` folder

To train on cluster:\
`
launch_job/dmc/cluster_sac.sh --experiment_name test_run_dog-walk --first_seed 0 --last_seed 0 --disable_wandb
`


[jax_badge_link]: https://tinyurl.com/5n8m53cy
