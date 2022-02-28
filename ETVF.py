from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import TVF_reference_args

ROLLOUT_SIZE = 128*128
ATARI_3_VAL = ['berzerk', 'boxing', 'zaxxon']
ATARI_5_VAL = ['Bowling', 'Qbert', 'Berzerk', 'Boxing', 'Zaxxon']

# updated args tuned for hard mode
TVF_HARD_ARGS = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'architecture': 'dual',
    'export_video': False,
    'epochs': 50,
    'use_compression': False,
    'warmup_period': 1000,
    'disable_ev': False,
    'seed': 0,
    'mutex_key': "DEVICE",

    # hard mode
    "terminal_on_loss_of_life": False,
    "reward_clipping": "off",
    "full_action_space": True,
    "repeat_action_probability": 0.25,

    'max_grad_norm': 5.0,
    'agents': 128,
    'n_steps': 128,
    'policy_mini_batch_size': 2048,
    'value_mini_batch_size': 512,
    'distil_mini_batch_size': 512,
    'policy_epochs': 3,
    'value_epochs': 2,
    'ppo_epsilon': 0.1,
    'policy_lr': 2.5e-4,
    'value_lr': 2.5e-4,
    'entropy_bonus': 1e-2, # increased entropy due to full action space
    'tvf_force_ext_value_distil': False,
    'hidden_units': 256,
    'gae_lambda': 0.95,
    'td_lambda': 0.95,

    # tvf params
    'use_tvf': True,
    'tvf_value_distribution': 'fixed_geometric',
    'tvf_horizon_distribution': 'fixed_geometric',
    'tvf_horizon_scale': 'log',
    'tvf_time_scale': 'log',
    'tvf_hidden_units': 256,
    'tvf_value_samples': 128,
    'tvf_horizon_samples': 32,
    'tvf_return_mode': 'exponential',
    'tvf_return_n_step': 20,
    'tvf_coef': 0.5,

    # distil / replay buffer (This would have been called h11 before
    'distil_epochs': 1,
    'distil_period': 1,
    'replay_size':   ROLLOUT_SIZE,
    'distil_batch_size': ROLLOUT_SIZE,
    'distil_beta': 1.0,
    'distil_lr': 2.5e-4,
    'replay_mode': "uniform",

    'dna_dual_constraint': 0,

    # horizon
    'gamma': 0.99997,
    'tvf_gamma': 0.99997,
    'tvf_max_horizon': 30000,

    # other
    'observation_normalization': True, # pong (and others maybe) do not work without this, so jsut default it to on..

    'hostname': '',
}


def tvf_ev(priority: int = 0):

    # experiment 1: initial training
    env = "Zaxxon"
    for seed in [1]:
        for samples in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            add_job(
                "TVF_EV1",
                run_name=f"game={env} samples={samples} ({seed})",
                env_name=env,
                seed=seed,
                epochs=50,
                tvf_value_samples=samples,
                tvf_horizon_samples=samples,
                tvf_value_distribution='fixed_geometric',
                tvf_horizon_distribution='fixed_geometric',
                priority=priority,
                default_params=TVF_HARD_ARGS,
            )

    # experiment 2: value learning (influence of samples)
    for seed in [1]:
        for samples in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            add_job(
                "TVF_EV2",
                run_name=f"game={env} samples={samples} ({seed})",
                env_name=env,
                seed=seed,
                epochs=10,
                policy_epochs=0,
                distil_epochs=0,
                freeze_observation_normalization=True,
                abs_mode="shadow", # so I can track noise
                tvf_value_samples=samples,
                tvf_horizon_samples=samples,
                reference_policy="../reference.pt.gz",

                tvf_value_distribution='fixed_geometric',
                tvf_horizon_distribution='fixed_geometric',
                priority=priority,
                default_params=TVF_HARD_ARGS,
            )

    # check sampling distribution
    pass

    # check n_steps
    pass

    # check other estimators (n_step, fast mc?, exp)
    pass

    # check others
    # log vs linear interpolation (doesn't matter I don't think)
    pass

    # sampling impact on training time
    # just benchmark for this..



def setup(priority_modifier=0):
    tvf_ev(300)
