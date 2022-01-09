"""
Atari-57 Main runs

"""

from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import PPO_reference_args, DNA_reference_args, TVF_reference_args
from runner_tools import ROLLOUT_SIZE, ATARI_57

def atari57(priority: int = 0):
    for env in ATARI_57:
        add_job(
            f"A57",
            env_name=env,
            run_name=f"game={env} rp1u (seed=1)",
            default_params=TVF_reference_args,
            distil_epochs=1,
            distil_period=1,
            replay_size=1 * ROLLOUT_SIZE,
            distil_batch_size=1 * ROLLOUT_SIZE,
            replay_mode="uniform",
            priority=priority,
            seed=1,
            hostname="ML",
        )
        add_job(
            f"A57_PPO",
            env_name=env,
            run_name=f"game={env} ppo_norm (seed=1)",
            default_params=PPO_reference_args,
            priority=priority,
            seed=1,
            hostname="desktop",
        )

def crazy_climber(priority: int = 0):
    # quick tests to see if we can fix crazy climber with full curve learning
    EPOCHS = 10 # just interested in ev for small horizons
    add_job(
        f"CC",
        env_name="CrazyClimber",
        run_name=f"ppo_norm (seed=1)",
        default_params=PPO_reference_args,
        priority=priority,
        epochs=EPOCHS,
        seed=1,
        hostname="desktop",
    )
    add_job(
        f"CC",
        env_name="CrazyClimber",
        run_name=f"dna_norm (seed=1)",
        default_params=DNA_reference_args,
        priority=priority,
        epochs=EPOCHS,
        seed=1,
        hostname="desktop",
    )
    add_job(
        f"CC",
        env_name="CrazyClimber",
        run_name=f"tvf_norm (seed=1)",
        default_params=TVF_reference_args,
        priority=priority,
        epochs=EPOCHS,
        seed=1,
        hostname="desktop",
    )
    add_job(
        f"CC",
        env_name="CrazyClimber",
        run_name=f"rp1u (seed=1)",
        default_params=TVF_reference_args,
        distil_epochs=1,
        distil_period=1,
        replay_size=1 * ROLLOUT_SIZE,
        distil_batch_size=1 * ROLLOUT_SIZE,
        replay_mode="uniform",
        priority=priority,
        epochs=EPOCHS,
        seed=1,
        hostname="desktop",
    )
    add_job(
        f"CC",
        env_name="CrazyClimber",
        run_name=f"rp1u [enhanced] (seed=1)",
        default_params=TVF_reference_args,
        distil_epochs=1,
        distil_period=1,
        replay_size=1 * ROLLOUT_SIZE,
        distil_batch_size=1 * ROLLOUT_SIZE,

        tvf_exp_gamma=1.5,
        tvf_horizon_samples=128,
        value_mini_batch_size=1024,
        distil_mini_batch_size=1024,
        max_micro_batch_size=1024,

        replay_mode="uniform",
        priority=priority,
        epochs=EPOCHS,
        seed=1,
        hostname="desktop",
    )

    add_job(
        f"CC",
        env_name="CrazyClimber",
        run_name=f"rp1u [enhanced, easy] (seed=1)",
        default_params=TVF_reference_args,
        distil_epochs=1,
        distil_period=1,
        replay_size=1 * ROLLOUT_SIZE,
        distil_batch_size=1 * ROLLOUT_SIZE,

        terminal_on_loss_of_life=True,
        value_transform="sqrt",

        tvf_exp_gamma=1.5,
        tvf_horizon_samples=128,
        value_mini_batch_size=1024,
        distil_mini_batch_size=1024,
        max_micro_batch_size=1024,

        replay_mode="uniform",
        priority=priority,
        seed=1,
        hostname="desktop",
    )

    add_job(
        f"CC",
        env_name="CrazyClimber",
        run_name=f"rp1u [anneal] (seed=1)",
        default_params=TVF_reference_args,
        distil_epochs=1,
        distil_period=1,
        replay_size=1 * ROLLOUT_SIZE,
        distil_batch_size=1 * ROLLOUT_SIZE,

        distil_lr_anneal=True,

        replay_mode="uniform",
        priority=priority,
        epochs=EPOCHS,
        seed=1,
        hostname="desktop",
    )

    add_job(
        f"CC",
        env_name="CrazyClimber",
        run_name=f"rp1u [dc] (seed=1)",
        default_params=TVF_reference_args,
        distil_epochs=1,
        distil_period=1,
        replay_size=1 * ROLLOUT_SIZE,
        distil_batch_size=1 * ROLLOUT_SIZE,

        dna_dual_constraint = 0.3,

        replay_mode="uniform",
        priority=priority,
        epochs=EPOCHS,
        seed=1,
        hostname="desktop",
    )

    add_job(
        f"CC",
        env_name="CrazyClimber",
        run_name=f"rp1u [mvh] (seed=1)",
        default_params=TVF_reference_args,
        distil_epochs=1,
        distil_period=1,
        replay_size=1 * ROLLOUT_SIZE,
        distil_batch_size=1 * ROLLOUT_SIZE,

        tvf_n_dedicated_value_heads=16, # might not work..

        replay_mode="uniform",
        priority=priority,
        epochs=EPOCHS,
        seed=1,
        hostname="desktop",
    )

    add_job(
        f"CC",
        env_name="CrazyClimber",
        run_name=f"rp1u [av_r] (seed=1)",
        default_params=TVF_reference_args,
        distil_epochs=1,
        distil_period=1,
        replay_size=1 * ROLLOUT_SIZE,
        distil_batch_size=1 * ROLLOUT_SIZE,

        tvf_value_scale_fn="linear", # average reward

        replay_mode="uniform",
        priority=priority,
        epochs=EPOCHS,
        seed=1,
        hostname="desktop",
    )


def setup(priority_modifier=0):
    # Initial experiments to make sure code it working, and find reasonable range for the hyperparameters.
    atari57(priority=priority_modifier)
    crazy_climber(priority=10)
