from runner_tools import WORKERS, add_job, random_search, Categorical, ATARI_57
import numpy as np

from typing import Union

QUICK_CHECK = False # limit to 1 seed on one environment with 0.1 epochs (just for testing)

ROLLOUT_SIZE = 128*128
ATARI_3_VAL = ['Assault', 'MsPacman', 'YarsRevenge']
ATARI_1_VAL = ['Assault']
ATARI_5 = ['BattleZone', 'DoubleDunk', 'NameThisGame', 'Phoenix', 'Qbert']

SEED_PENALITY = 25 # how much to deprioritise seeds


TVF5_ARGS = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'hostname': '',
    'architecture': 'dual',
    'epochs': 50,
    'obs_compression': False,
    'upload_batch': True,  # much faster
    'warmup_period': 1000,
    'disable_ev': False,
    'seed': 0,
    'mutex_key': "DEVICE",

    'max_micro_batch_size': 1024,   # might help now that we upload the entire batch?

    'max_grad_norm': 5.0,
    'agents': 128,                  # HPS
    'n_steps': 128,                 # HPS
    'policy_mini_batch_size': 2048, # Trying larger
    'value_mini_batch_size': 512,   # should be 256, but 512 for performance (maybe needs to be larger for tvf?)
    'distil_mini_batch_size': 512,  #
    'policy_epochs': 2,             # reasonable guess
    'value_epochs': 1,              # reasonable guess
    'distil_epochs': 1,             # reasonable guess
    'ppo_epsilon': 0.2,             # allows faster policy movement
    'policy_lr': 2.5e-4,
    'value_lr': 2.5e-4,
    'distil_lr': 2.5e-4,
    'entropy_bonus': 1e-2,          # standard
    'hidden_units': 512,            # standard

    'lambda_policy': 0.8,
    'lambda_value': 0.95,

    # tvf
    'use_tvf': True,
    'tvf_coef': 10,                 # 10 might be too much
    'tvf_return_n_step': 20,        # should be 20 maybe, or higher maybe?
    'tvf_return_samples': 4,        # 4 samples is much faster
    'tvf_value_heads': 128,         # maybe need more?
    'tvf_horizon_trimming': 'average',
    'tvf_return_mode': "advanced2",

    # noise is on by default
    'use_sns': True,
    'sns_max_heads': 7,
    'sns_period': 3,
    'ag_sns_min_h': 30,             # give model a bit more room
    'ag_sns_max_h': 10000,
    "ag_sns_threshold": 12.5,        # this is about right

    # by default work out auto_horizon in background, but do not enable
    'ag_mode': 'sns',               # but not enabled
    "ag_sns_delay": int(5e6),       # best to wait awhile before modifying gamma, maybe even 10M?

    # stuck
    'repeated_action_penalty': 0.01,
    'max_repeated_actions': 50,

    # distil, but no replay buffer
    'replay_size': 0,
    'distil_period': 2,             # not sure if this is right?
    'distil_batch_size': 1*128*128,
    'distil_beta': 1.0,
    'distil_max_heads': -1,         # no idea about this parameter...


    # horizon
    'gamma': 0.99997,
    'tvf_gamma': 0.99997,
    'tvf_max_horizon': 30000,

    # other
    'observation_normalization': True, # pong (and others maybe) do not work without this, so jsut default it to on..
}


def add_run(experiment: str, run_name: str, default_args, env_args, subset:list, seeds:Union[int, list]=3, priority=0, seed_params=None, **kwargs):

    args = default_args.copy()
    args.update(env_args)

    if seed_params is None:
        seed_params = {}

    if type(seeds) is int:
        seeds = list(range(1, seeds+1))

    if QUICK_CHECK:
        # just for testing
        seed = 1
        env = subset[0]
        add_job(
            experiment,
            run_name=f"game={env} {run_name} ({seed})",
            env_name=env,
            seed=seed,
            priority=priority - ((seed - 1) * SEED_PENALITY),
            default_params=args,
            epochs=0.1,
            **seed_params.get(seed, {}),
            **kwargs,
        )
        return

    for seed in seeds:
        for env in subset:
            add_job(
                experiment,
                run_name=f"game={env} {run_name} ({seed})",
                env_name=env,
                seed=seed,
                priority=priority - ((seed - 1) * SEED_PENALITY),
                default_params=args,
                **seed_params.get(seed, {}),
                **kwargs,
            )


HARD_MODE_ARGS = {
    # hard mode
    "terminal_on_loss_of_life": False,
    "reward_clipping": "off",
    "full_action_space": True,
    "repeat_action_probability": 0.25,
}

EASY_MODE_ARGS = {
    # hard mode
    "terminal_on_loss_of_life": True,
    "reward_clipping": "off",
    "full_action_space": False,
    "repeat_action_probability": 0.0,
}

# ----------------------------------------------------------------------------------------------------------
# TVF 5

def tvf5_auto(priority: int = 0):

    # really just want to know if this works or not...

    COMMON_ARGS = {
        'seeds': 2,
        'priority': priority,
        'env_args': HARD_MODE_ARGS,
        'experiment': "TVF5_AUTO",
        'default_args': TVF5_ARGS,
        'epochs': 50,  # 20 is enough for these two games
    }

    add_run(
        run_name=f"auto",
        subset=["CrazyClimber", "Skiing", "Zaxxon", "Surround"] + ATARI_3_VAL,
        override_reward_normalization_gamma=0.999, # compromise
        use_ag=True,
        **COMMON_ARGS,
    )


def tvf5_tuning(priority: int = 0):

    # testing out a few ideas...
    COMMON_ARGS = {
        'seeds': 3,
        'priority': priority,
        'env_args': HARD_MODE_ARGS,
        'experiment': "TVF5_TUNING",
        'default_args': TVF5_ARGS,
        'epochs': 50,  # 20 is enough for these two games
        'subset': ATARI_3_VAL,
         # put these on the cluster so they actually get done...
         'hostname': "cluster",
         'device': 'cuda',
    }

    add_run(
        run_name=f"ref",
        **COMMON_ARGS,
    )

    # can we make things more consistent with tvl?
    for tvl in [0.3, 1.0, 3.0]:
        add_run(
            run_name=f"ref",
            distil_loss_value_target=tvl,
            **COMMON_ARGS,
        )

    # try distil settings
    add_run(
        run_name=f"distil off",
        distil_epochs=0,
        **COMMON_ARGS,
    )
    add_run(
        run_name=f"distil before",
        distil_order="before_policy",
        **COMMON_ARGS,
    )
    add_run( # default is 1_every_2
        run_name=f"distil 2_every_1",
        distil_period=1,
        **COMMON_ARGS,
    )
    add_run(
        run_name=f"distil 1_every_2",
        distil_epochs=2,
        **COMMON_ARGS,
    )

    # check if our estimator is causing problems (advanced seems like it might be good?)
    for mode in ["exponential", "advanced", "fixed"]:
        add_run(
            run_name=f"tvf_return_mode={mode}",
            tvf_return_mode=mode,
            **COMMON_ARGS,
        )




def setup():

    tvf5_auto()
    tvf5_tuning()
    pass
