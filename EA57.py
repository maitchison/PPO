"""
Atari-57 Main runs

"""

from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import ROLLOUT_SIZE, ATARI_57, HARD_MODE, EASY_MODE, RAINBOW_MODE
from runner_tools import __PPO_reference_args, __DNA_reference_args, __TVF_reference_args, __TVF99_reference_args, \
    __ERP_reference_args, __RP1U_reference_args, PPO_reference_args, TVF_reference_args, RP1U_reference_args


def atari57_rainbow_settings(priority: int = 0):
    for env in ATARI_57:
        COMMON_ARGS = {
            'env_name': env,
            'priority': priority,
            'seed': 1,
            'hostname': "ML",
            'epochs': 50,
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


def atari57_hard_settings(priority: int = 0):

    ATARI_5 = ('Alien', 'Breakout', 'BankHeist', 'BeamRider', 'Amidar', 'Assault', 'Gravitar')

    N_STEPS = 512  # helps with stability
    AGENTS = 128
    ROLLOUT_SIZE = N_STEPS * AGENTS

    UPGRADED_ARGS = {
        'n_steps': 512,
        'agents': 128,
        'policy_mini_batch_size': 2048,  # helps with stability
        'entropy_scaling': True,  # handles changes in stocasticity better.
        'tvf_value_distribution': 'saturated_geometric',
        'tvf_horizon_distribution': 'saturated_geometric',  # pay more attention to short horizons
        'tvf_horizon_samples': 64,  # more samples gives better ev
        'tvf_return_n_step': 20,  # this is much lower than I expected

        'anneal_target_epoch': 50,

        # new replay setting for larger buffer...
        'distil_epochs': 1,
        'distil_period': 1,
        'replay_size': 1 * ROLLOUT_SIZE,
        'distil_batch_size': 1 * ROLLOUT_SIZE,
        'replay_mode': "uniform",

    }

    for env in ATARI_5:

        COMMON_ARGS = {
            'env_name': env,
            'seed': 1,
            'hostname': "ML",
            'epochs': 50,
        }

        COMMON_ARGS.update(HARD_MODE)
        COMMON_ARGS.update(UPGRADED_ARGS)

        add_job(
            f"A57_HARD",
            run_name=f"game={env} ppo (1)",
            default_params=PPO_reference_args,
            priority=priority,
            **COMMON_ARGS,
        )

        add_job(
            f"A57_HARD",
            run_name=f"game={env} rp1u (1)",
            default_params=RP1U_reference_args,
            priority=priority,
            **COMMON_ARGS,
        )

        add_job(
            f"A57_HARD",
            run_name=f"game={env} rp1u+rnd (1)",
            default_params=RP1U_reference_args,
            use_rnd=True,
            ir_anneal='linear',
            ir_scale=1.0,
            priority=priority-10,
            **COMMON_ARGS,
        )



def setup(priority_modifier=0):
    # Initial experiments to make sure code it working, and find reasonable range for the hyperparameters.
    atari57_rainbow_settings(priority=priority_modifier)
    atari57_hard_settings(priority=priority_modifier)

