"""
Atari-57 Main runs

"""

from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import ROLLOUT_SIZE, ATARI_57, HARD_MODE, EASY_MODE, RAINBOW_MODE
from runner_tools import __PPO_reference_args, __DNA_reference_args, __TVF_reference_args, __TVF99_reference_args, \
    __ERP_reference_args, __RP1U_reference_args


def atari57(priority: int = 0):
    for env in ATARI_57:
        COMMON_ARGS = {
            'env_name': env,
            'priority': priority,
            'seed': 1,
            'hostname': "ML",
            'epochs': 40,
        }

        COMMON_ARGS.update(RAINBOW_MODE)

        add_job(
            f"A57_RAINBOW",
            run_name=f"game={env} tvf (1)",
            default_params=__TVF_reference_args,
            **COMMON_ARGS,
        )
        add_job(
            f"A57_RAINBOW",
            run_name=f"game={env} dna (1)",
            default_params=__DNA_reference_args,
            **COMMON_ARGS,
        )
        add_job(
            f"A57_RAINBOW",
            run_name=f"game={env} ppo (1)",
            default_params=__PPO_reference_args,
            **COMMON_ARGS,
        )

        # new folder as code for exploration was added after the others
        # started...
        # would be nice to go back and get the others all running on the same codebase.
        add_job(
            f"A57_RAINBOW_2",
            run_name=f"game={env} erp (1)",
            default_params=__ERP_reference_args,
            **COMMON_ARGS,
        )


def setup(priority_modifier=0):
    # Initial experiments to make sure code it working, and find reasonable range for the hyperparameters.
    atari57(priority=priority_modifier)
    pass
