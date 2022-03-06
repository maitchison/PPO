"""
Exploration experiments

"""

from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import RP1U_reference_args
from runner_tools import ROLLOUT_SIZE, ATARI_57, HARD_MODE, EASY_MODE, RAINBOW_MODE, ATARI_57, PPO_reference_args, \
    TVF_reference_args, DNA_reference_args


def setup(priority_modifier=0):

    for env in ['Breakout', 'CrazyClimber', 'MontezumaRevenge']:

        seed = 1

        COMMON_ARGS = {
            'env_name': env,
            'seed': seed,
            'hostname': "",
            'epochs': 10,
            'seed': 1,
            'priority': priority_modifier,
        }

        PPO_ARGS = PPO_reference_args.copy()
        PPO_ARGS.update(HARD_MODE)
        PPO_ARGS.update(COMMON_ARGS)

        DNA_ARGS = DNA_reference_args
        DNA_ARGS.update(HARD_MODE)
        DNA_ARGS.update(COMMON_ARGS)

        TVF_ARGS = RP1U_reference_args.copy()
        TVF_ARGS.update(HARD_MODE)
        TVF_ARGS.update(COMMON_ARGS)

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} ppo default ({seed})",
            default_params=PPO_ARGS,
        )

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} dna default ({seed})",
            default_params=DNA_ARGS,
        )

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} tvf default ({seed})",
            default_params=TVF_ARGS,
        )

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} tvf easy ({seed})",
            terminal_on_loss_of_life=False,
            reward_clipping="off",
            full_action_space=False,
            repeat_action_probability=0.0,
            default_params=TVF_ARGS,
        )

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} tvf reduced_actions ({seed})",
            full_action_space=False,
            default_params=TVF_ARGS,
        )

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} tvf entropy_scaling ({seed})",
            entropy_scaling=True,
            default_params=TVF_ARGS,
        )

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} tvf eb=0.01 ({seed})",
            entropy_bonus=0.01,
            default_params=TVF_ARGS,
        )

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} tvf eb=0.03 ({seed})",
            entropy_bonus=0.03,
            default_params=TVF_ARGS,
        )

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} tvf gamma=0.99 ({seed})",
            gamma=0.99,
            tvf_gamma=0.99,
            default_params=TVF_ARGS,
        )

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} tvf no_distil ({seed})",
            distil_epochs=0,
            default_params=TVF_ARGS,
        )

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} tvf no_replay ({seed})",
            replay_size=0,
            default_params=TVF_ARGS,
        )

        add_job(
            experiment_name="STUCK",
            run_name=f"{env} tvf simple_distil ({seed})",
            tvf_force_ext_value_distil=True,
            default_params=TVF_ARGS,
        )

    for seed in range(3):
        # try applying a penality
        env = "MontezumaRevenge"
        TVF_ARGS['env_name'] = env
        TVF_ARGS['seed'] = seed
        for rap in [0, 0.25, 1.0]:
            add_job(
                experiment_name="STUCK_2",
                run_name=f"{env} tvf rap={rap} ({seed})",
                repeated_action_penalty=rap,
                default_params=TVF_ARGS,
            )
