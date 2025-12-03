import os
import shutil
import subprocess
import unittest


class TestMuJoCo(unittest.TestCase):
    def test_sac(self):
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../experiments/mujoco/exp_output/_test_sac_HalfCheetah"
        )
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        returncode = subprocess.run(
            [
                "python3",
                "experiments/mujoco/sac.py",
                "--experiment_name",
                "_test_sac_HalfCheetah",
                "--seed",
                "1",
                "--disable_wandb",
                "--features_q",
                "2",
                "3",
                "--features_pi",
                "2",
                "3",
                "--replay_buffer_capacity",
                "100",
                "--batch_size",
                "3",
                "--update_horizon",
                "1",
                "--gamma",
                "0.99",
                "--learning_rate",
                "1e-4",
                "--horizon",
                "10",
                "--n_samples",
                "10",
                "--tau",
                "0.5",
                "--n_initial_samples",
                "5",
            ]
        ).returncode
        assert returncode == 0, "The command should not have raised an error."

        shutil.rmtree(save_path)
