from runner_tools import WORKERS, MYSTIC_FIVE, add_job

ROLLOUT_SIZE = 128*128

E2_args = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'architecture': 'dual',
    'export_video': False,
    'epochs': 50,
    'use_compression': 'auto',
    'warmup_period': 1000,
    'disable_ev': False,
    'seed': 0,
    'use_mutex': True,

    # env parameters
    'time_aware': True,
    'terminal_on_loss_of_life': False,
    'reward_clipping': "off",
    'value_transform': 'identity',

    'max_grad_norm': 5.0,
    'agents': 128,
    'n_steps': 128,
    'policy_mini_batch_size': 512,
    'value_mini_batch_size': 512,
    'policy_epochs': 3,
    'value_epochs': 2,
    'ppo_epsilon': 0.1,
    'policy_lr': 2.5e-4,
    'value_lr': 2.5e-4,
    'entropy_bonus': 1e-3,
    'tvf_force_ext_value_distil': False,
    'hidden_units': 256,
    'gae_lambda': 0.95,

    # tvf params
    'use_tvf': True,
    'tvf_value_distribution': 'fixed_geometric',
    'tvf_horizon_distribution': 'fixed_geometric',
    'tvf_horizon_scale': 'log',
    'tvf_time_scale': 'log',
    'tvf_hidden_units': 256,
    'tvf_value_samples': 128,
    'tvf_horizon_samples': 32,
    'tvf_mode': 'exponential',
    'tvf_exp_gamma': 2.0,
    'tvf_coef': 0.5,
    'tvf_soft_anchor': 0,
    'tvf_exp_mode': "transformed",

    # distil / replay buffer (This would have been called h11 before
    'distil_epochs': 1,
    'distil_period': 1,
    'replay_size':   1*ROLLOUT_SIZE,
    'distil_resampling': False,
    'distil_batch_size': ROLLOUT_SIZE//2,
    'distil_beta': 1.0,
    'distil_lr': 2.5e-4,
    'replay_mode': "uniform",
    'replay_mixing': False,
    'dna_dual_constraint': 0.3,

    # horizon
    'gamma': 0.99997,
    'tvf_gamma': 0.99997,
    'tvf_max_horizon': 30000,

    'hostname': "ML-Rig", # for reproducibility
}

for env in MYSTIC_FIVE:
    for replay_size in [0, 1, 2, 4, 8, 16, 32]:
        add_job(
            f"E2_ReplaySize",
            env_name=env,
            run_name=f"game={env} rs={replay_size}",
            replay_size=replay_size*ROLLOUT_SIZE,
            default_params=E2_args,
            use_compression=True,
            priority=200,
        )
