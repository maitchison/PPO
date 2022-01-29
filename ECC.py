def crazy_climber(priority: int = 0):
    # quick tests to see if we can fix crazy climber with full curve learning
    EPOCHS = 30 # just interested in ev for small horizons
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
        run_name=f"rp1u [enhanced, easy2] (seed=1)",
        default_params=TVF_reference_args,
        distil_epochs=1,
        distil_period=1,
        replay_size=1 * ROLLOUT_SIZE,
        distil_batch_size=1 * ROLLOUT_SIZE,

        terminal_on_loss_of_life=True,

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
        epochs=50,
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

        tvf_n_dedicated_value_heads=16, # old code... might not work..

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
