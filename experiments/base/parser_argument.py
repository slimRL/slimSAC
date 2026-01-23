import argparse
from functools import wraps
from typing import Callable, List


def output_added_arguments(add_algo_arguments: Callable) -> Callable:
    @wraps(add_algo_arguments)
    def decorated(parser: argparse.ArgumentParser) -> List[str]:
        unfiltered_old_arguments = list(parser._option_string_actions.keys())

        add_algo_arguments(parser)

        unfiltered_arguments = list(parser._option_string_actions.keys())
        unfiltered_added_arguments = [
            argument for argument in unfiltered_arguments if argument not in unfiltered_old_arguments
        ]

        return [
            argument.strip("-")
            for argument in unfiltered_added_arguments
            if argument.startswith("--") and argument not in ["--help"]
        ]

    return decorated


@output_added_arguments
def add_base_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed of the experiment.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-dw",
        "--disable_wandb",
        help="Disable wandb.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-rbc",
        "--replay_buffer_capacity",
        help="Replay Buffer capacity.",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Batch size for training.",
        type=int,
        default=256,
    )
    parser.add_argument(
        "-nis",
        "--n_initial_samples",
        help="Number of initial samples before the training starts.",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "-uh",
        "--update_horizon",
        help="Value of n in n-step TD update.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        help="Discounting factor.",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate.",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "-horizon",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        help="Total number of collected samples.",
        default=1_000_000,
    )
    parser.add_argument(
        "-at",
        "--architecture_type",
        help="Type of architecture.",
        type=str,
        default="fc",
        choices=["fc", "simbav1"],
    )
    parser.add_argument(
        "-fq",
        "--features_q",
        type=int,
        nargs="*",
        help="List of features for the Q-networks.",
        default=[256, 256],
    )
    parser.add_argument(
        "-fpi",
        "--features_pi",
        type=int,
        nargs="*",
        help="List of features for the policy network.",
        default=[256, 256],
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        help="Weighting of the regularization in weight decay.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-tau",
        "--tau",
        help="Soft target update parameter.",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--double_q",
        help="Whether to combat overestimation with a twin Q-network.",
        default=False,
        action="store_true",
    )


@output_added_arguments
def add_sac_arguments(parser: argparse.ArgumentParser):
    pass
