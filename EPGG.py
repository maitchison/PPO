from runner_tools import WORKERS, add_job, random_search, Categorical, DIVERSE_10
from runner_tools import PPO_reference_args, DNA_reference_args, TVF_reference_args

ROLLOUT_SIZE = 128*128

# if this works try undiscounted :)

def normed_runs(priority:int=0):
    """
    Additional runs with normalization
    """
    HOSTNAME = "ML"
    for env in DIVERSE_10:
        add_job(
            f"PGG_PerGameGamma",
            env_name=env,
            run_name=f"game={env} ppo_norm (seed=1)",
            default_params=PPO_reference_args,
            priority=priority,
            use_compression=False, # not needed
            seed=1,
            hostname=HOSTNAME,
        )
        add_job(
            f"PGG_PerGameGamma",
            env_name=env,
            run_name=f"game={env} dna_norm (seed=1)",
            default_params=DNA_reference_args,
            priority=priority,
            use_compression=False,  # not needed
            seed=1,
            hostname=HOSTNAME,
        )
        add_job(
            f"PGG_PerGameGamma",
            env_name=env,
            run_name=f"game={env} tvf_norm (seed=1)",
            default_params=TVF_reference_args,
            priority=priority,
            use_compression=False,  # not needed
            seed=1,
            hostname=HOSTNAME,
        )
        add_job(
            f"PGG_PerGameGamma",
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
            hostname=HOSTNAME,
        )

        add_job(
            f"PGG_PerGameGamma",
            env_name=env,
            run_name=f"game={env} rp16u (seed=1)",
            default_params=TVF_reference_args,

            distil_epochs=1,
            distil_period=8,
            replay_size=16 * ROLLOUT_SIZE,
            distil_batch_size=16 * ROLLOUT_SIZE,
            replay_mode="uniform",

            priority=priority,
            seed=1,
            hostname=HOSTNAME,
        )

        add_job(
            f"PGG_PerGameGamma",
            env_name=env,
            run_name=f"game={env} rp16s (seed=1)",
            default_params=TVF_reference_args,

            distil_epochs=1,
            distil_period=8,
            replay_size=16 * ROLLOUT_SIZE,
            distil_batch_size=16 * ROLLOUT_SIZE,
            replay_mode="sequential",

            priority=priority,
            seed=1,
            hostname=HOSTNAME,
        )


def setup(priority_modifier=0):
    normed_runs(priority=priority_modifier)
