from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import TVF_reference_args

ROLLOUT_SIZE = 128*128
ATARI_3_VAL = ['NameThisGame', 'WizardOfWor', 'YarsRevenge']

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
    'tvf_horizon_samples': 128,
    'tvf_return_mode': 'exponential',
    'tvf_return_n_step': 40,
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


    EV_SETTINGS = TVF_HARD_ARGS.copy()
    EV_SETTINGS['hostname'] = "ML"
    EV_SETTINGS['priority'] = priority

    # step 1: generate a reference policy for each game in the validation set
    # zaxxon is just to make sure we did not regress
    for env in ATARI_3_VAL + ['Zaxxon']:
        add_job(
            "TVF_EV1_DYNAMIC",
            run_name=f"game={env} reference",
            env_name=env,
            seed=1,
            epochs=50,
            save_model_interval=1,
            default_params=EV_SETTINGS,
        )

    EV2_SETTINGS = EV_SETTINGS.copy()
    EV2_SETTINGS.update({
        'epochs': 10,
        'policy_epochs': 0,
        'distil_epochs': 0,
        'freeze_observation_normalization': True,
        'abs_mode': "shadow",  # so I can track noise
        'checkpoint_every': int(1e6),  # this will allow me to take true value ev estimates.
        'warmup_period': 0,  # not needed and will skew scores early on...
        'priority': 100,
    })

    # use fixed reward scales rather than renormalizing. If we renormalize we could be off by 20% or so and would
    # have to adjust for that. This way we can just use the normalized value estimates, and should be able to combine
    # them more easily too. (although normalizing the error sounds like a good idea).
    fixed_reward_scale = {
        # taken from reference runs
        #'Breakout': 1/90.872, # <- change to name this game
        'Zaxxon': 1/5362.42,
        'NameThisGame': 1/3082.216567427429,
        'WizardOfWor': 1/4307.944,
        'YarsRevenge': 1/23021.322,
    }

    # experiment 2: value learning (sampling and sample distribution)
    # for env in ATARI_3_VAL:
    #     for seed in [1, 2, 3]:
    #         for distribution in ['fixed_geometric', 'geometric', 'fixed_linear', 'linear', 'saturated_fixed_geometric', 'saturated_geometric']:
    #             for samples in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    #                 add_job(
    #                     "TVF_EV2",
    #                     run_name=f"game={env} distribution={distribution} samples={samples} ({seed})",
    #                     reference_policy=f"../game={env} reference.pt.gz",
    #                     fixed_reward_scale=...
    #                     env_name=env,
    #                     seed=seed,
    #
    #                     tvf_value_samples=samples,
    #                     tvf_horizon_samples=samples,
    #                     tvf_value_distribution=distribution,
    #                     tvf_horizon_distribution=distribution,
    #
    #                     default_params=EV2_SETTINGS,
    #                 )

    # experiment 3: value learning (influence of return estimator)
    for env in ATARI_3_VAL + ['Zaxxon']:
        if env not in fixed_reward_scale:
            continue
        for seed in [1, 2, 3]:
            for return_mode in ['exponential', 'fixed', 'hyperbolic']: #, 'quadratic']:
                for n_step in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                    add_job(
                        "TVF_EV3",
                        run_name=f"game={env} return_mode={return_mode} n_step={n_step} ({seed})",
                        n_steps=max(n_step, 128) if return_mode in ["fixed", "exponential"] else 128, # extend n_steps for long n_steps
                        reward_scale=fixed_reward_scale[env],
                        reward_normalization=False,
                        reference_policy=f"../{env}.pt.gz",
                        env_name=env,
                        seed=seed,

                        tvf_return_mode=return_mode,
                        tvf_return_n_step=n_step,

                        default_params=EV2_SETTINGS,
                    )
            # special case of uniform
            add_job(
                "TVF_EV3",
                run_name=f"game={env} return_mode=uniform ({seed})",
                reference_policy=f"../{env}.pt.gz",
                reward_scale=fixed_reward_scale[env],
                reward_normalization=False,
                env_name=env,

                tvf_return_mode='uniform',
                tvf_return_n_step=1,

                default_params=EV2_SETTINGS,
            )


def setup(priority_modifier=0):
    tvf_ev(500)
    pass
