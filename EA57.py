"""
Atari-57 Main runs

"""

from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import ROLLOUT_SIZE, ATARI_57, HARD_MODE, EASY_MODE, RAINBOW_MODE
from runner_tools import PPO_reference_args, DNA_reference_args, TVF_reference_args, TVF99_reference_args, \
    ERP_reference_args, RP1U_reference_args


def atari57(priority: int = 0):
    for env in ATARI_57:
        COMMON_ARGS = {
            'env_name': env,
            'priority': priority,
            'seed': 1,
            'hostname': "ML",
            'epochs': 30, # just to get a feel for things early on
        }

        COMMON_ARGS.update(RAINBOW_MODE)

        add_job(
            f"A57_RAINBOW",
            run_name=f"game={env} tvf (1)",
            default_params=TVF_reference_args,
            **COMMON_ARGS,
        )
        add_job(
            f"A57_RAINBOW",
            run_name=f"game={env} dna (1)",
            default_params=DNA_reference_args,
            **COMMON_ARGS,
        )
        add_job(
            f"A57_RAINBOW",
            run_name=f"game={env} ppo (1)",
            default_params=PPO_reference_args,
            **COMMON_ARGS,
        )

        # new folder as code for exploration was added after the others
        # started...
        # would be nice to go back and get the others all running on the same codebase.
        add_job(
            f"A57_RAINBOW_2",
            run_name=f"game={env} erp (1)",
            default_params=ERP_reference_args,
            **COMMON_ARGS,
        )


def setup(priority_modifier=0):
    # Initial experiments to make sure code it working, and find reasonable range for the hyperparameters.
    atari57(priority=priority_modifier)
    pass
