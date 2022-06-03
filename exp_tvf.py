from runner_tools import WORKERS, add_job, random_search, Categorical, ATARI_57
import numpy as np

from typing import Union

QUICK_CHECK = False # limit to 1 seed on one environment with 0.1 epochs (just for testing)

ROLLOUT_SIZE = 128*128
ATARI_3_VAL = ['Assault', 'MsPacman', 'YarsRevenge']
ATARI_1_VAL = ['Assault']
ATARI_5 = ['BattleZone', 'DoubleDunk', 'NameThisGame', 'Phoenix', 'Qbert']


"""
Todo:
 [x] check if tvf oss differs from value loss, maybe beta needs tuning? (this seems fine...)
 [ ] add replay back in
 [ ] revert back to 512 units (from 256)
 [x] make sure gae_lambda and td_lambda are all good (they are)
 
 bonus ideas:
  - replay?
  - simplified distil?

"""

def add_run(experiment: str, run_name: str, default_args, env_args, subset:list, seeds:Union[int, list]=3, priority=0, seed_params=None, **kwargs):

    args = HPS_ARGS.copy()
    args.update(default_args)
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
            priority=priority - ((seed - 1) * 100),
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
                priority=priority - ((seed - 1) * 100),
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

# These are the best settings from the HPS, but not from the axis search performed later.
HPS_ARGS = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'hostname': '',
    'architecture': 'dual',
    'export_video': False,
    'epochs': 50,
    'use_compression': False,
    'upload_batch': True,  # much faster
    'warmup_period': 1000,
    'disable_ev': False,
    'seed': 0,
    'mutex_key': "DEVICE",

    'max_grad_norm': 5.0,
    'agents': 128,                  # HPS
    'n_steps': 128,                 # HPS
    'policy_mini_batch_size': 2048, # HPS
    'value_mini_batch_size': 512,   # should be 256, but 512 for performance
    'distil_mini_batch_size': 512,  # should be 256, but 512 for performance
    'policy_epochs': 2,             # reasonable guess
    'value_epochs': 2,              # reasonable guess
    'distil_epochs': 2,             # reasonable guess
    'ppo_epsilon': 0.2,             # allows faster policy movement
    'policy_lr': 2.5e-4,
    'value_lr': 2.5e-4,
    'distil_lr': 2.5e-4,
    'entropy_bonus': 1e-2,           # standard
    'hidden_units': 512,             # standard
    'gae_lambda': 0.95,              # standard
    'td_lambda': 0.95,               # standard
    'repeated_action_penalty': 0.25, # HPS says 0, but I think we need this..

    # tvf params
    'use_tvf': False,

    # distil / replay buffer (This would have been called h11 before
    'distil_period': 1,
    'replay_size': 0,       # off for now...
    'distil_beta': 1.0,     # was 1.0

    'replay_mode': "uniform",

    # horizon
    'gamma': 0.999,

    # other
    'observation_normalization': True, # pong (and others maybe) do not work without this, so jsut default it to on..
}

# used in the PPO Paper
PPO_ORIG_ARGS = HPS_ARGS.copy()
PPO_ORIG_ARGS.update({
    'n_steps': 128,            # no change
    'agents': 8,
    'ppo_epsilon': 0.1,
    'policy_lr': 2.5e-4,
    'policy_lr_anneal': True,
    'ppo_epsilon_anneal': True,
    'entropy_bonus': 1e-2,     # no change
    'gamma': 0.99,
    'policy_epochs': 3,
    'td_lambda': 0.95,
    'gae_lambda': 0.95,
    'policy_mini_batch_size': 256,
    'vf_coef': 2.0, # because I use vf 0.5*MSE
    'value_epochs': 0,
    'distil_epochs': 0,
    'architecture': 'single',
})

DNA_TUNED_ARGS = HPS_ARGS.copy()
DNA_TUNED_ARGS.update({
    'gae_lambda': 0.8,
    'td_lambda': 0.95,
    'policy_epochs': 2,
    'value_epochs': 1,
    'distil_epochs': 2,
})

PPO_TUNED_ARGS = HPS_ARGS.copy()
PPO_TUNED_ARGS.update({
    'gae_lambda': 0.95,
    'td_lambda': 0.95,
    'policy_epochs': 1,
    'value_epochs': 0,
    'distil_epochs': 0,
    'architecture': 'single',
    'policy_network': 'nature_fat',
})

PPO_FAST_ARGS = HPS_ARGS.copy()
PPO_FAST_ARGS.update({
    'gae_lambda': 0.95,
    'td_lambda': 0.95,
    'policy_epochs': 1,
    'value_epochs': 0,
    'distil_epochs': 0,
    'architecture': 'single',
    'policy_network': 'nature',
})

PPG_ARGS = HPS_ARGS.copy()
PPG_ARGS.update({
    'policy_epochs': 1,
    'value_epochs': 1,
    'distil_epochs': 0,
    'aux_epochs': 6,
    'aux_target': 'vtarg',
    'aux_source': 'value',
    'aux_period': 32,
    'replay_mode': 'sequential',
    'replay_size': 32*128*128,  # this is 0.5M frames (might need more?)
    'distil_batch_size': 32*128*128, # use entire batch (but only every 32th step)
    'use_compression': True,
    'upload_batch': False,
})


# these are just my best guess and based on some initial experiments
TVF_INITIAL_ARGS = DNA_TUNED_ARGS.copy()
TVF_INITIAL_ARGS.update({
    'tvf_force_ext_value_distil': False,
    'hidden_units': 512,        # changed?
    'gae_lambda': 0.8,

    'policy_epochs': 2,
    'distil_epochs': 2,
    'value_epochs': 2,

    # tvf params
    'use_tvf': True,
    'tvf_mode': 'fixed',        # this is much better
    'tvf_hidden_units': 0,      # not needed / not wanted for fixed
    'tvf_value_distribution': 'fixed_geometric',
    'tvf_horizon_distribution': 'fixed_geometric',
    'tvf_horizon_scale': 'log',
    'tvf_time_scale': 'log',
    'tvf_value_samples': 128,   # probably too much!
    'tvf_horizon_samples': 128, # probably too much!
    'tvf_return_mode': 'exponential',
    'tvf_return_n_step': 20,    # should be 20 maybe, or higher maybe?
    'tvf_return_samples': 16,   # too low probably?
    'tvf_coef': 1.0,

    # yes please to replay, might remove later though
    'replay_size': 1 * 128 * 128,
    'distil_batch_size': 1 * 128 * 128,

    # horizon
    'gamma': 0.99997,
    'tvf_gamma': 0.99997,
    'tvf_max_horizon': 30000,
})


def merge_dict(a, b):
    x = a.copy()
    x.update(b)
    return x


def reference(priority: int = 0):

    COMMON_ARGS = {
        'seeds': 1,
        'subset': ATARI_3_VAL,
        'priority': priority,
        'hostname': "",
        'env_args': HARD_MODE_ARGS,
        'experiment': "REFERENCE",
    }

    # reference runs just to see how we're doing.

    add_run(
        run_name="ppo (reference)",
        default_args=PPO_TUNED_ARGS,
        **COMMON_ARGS,
    )

    add_run(
        run_name="dna (reference)",
        default_args=DNA_TUNED_ARGS,
        **COMMON_ARGS,
    )

    add_run(
        run_name="tvf (reference)",
        default_args=TVF_INITIAL_ARGS,
        **COMMON_ARGS
    )


def horizon(priority: int = 0):

    COMMON_ARGS = {
        'seeds': 1,
        'subset': ATARI_3_VAL,
        'priority': priority,
        'hostname': "",
        'env_args': HARD_MODE_ARGS,
        'default_args': TVF_INITIAL_ARGS,
        'experiment': "TVF_HORIZON",
    }

    # check horizons
    add_run(
        run_name="tvf (10k)",
        gamma=0.9999,
        tvf_gamma=0.9999,
        **COMMON_ARGS
    )

    add_run(
        run_name="tvf (1k)",
        gamma=0.999,
        tvf_gamma=0.999,
        **COMMON_ARGS
    )

    # check rediscounting
    add_run(
        run_name="tvf (30k_10k)",
        gamma=0.9999,
        tvf_gamma=0.99997,
        **COMMON_ARGS
    )

    add_run(
        run_name="tvf (30k_1k)",
        gamma=0.999,
        tvf_gamma=0.99997,
        **COMMON_ARGS
    )


def returns(priority: int = 0):

    COMMON_ARGS = {
        'seeds': 1,
        'subset': ATARI_3_VAL,
        'priority': priority,
        'hostname': "",
        'env_args': HARD_MODE_ARGS,
        'default_args': TVF_INITIAL_ARGS,
        'experiment': "TVF_RETURN",
    }

    # check n_steps and samples
    for n_step in [80]: # what I think it should be...
        for samples in [1, 4, 16, 64]:
            add_run(
                run_name=f"n_step={n_step} samples={samples}",
                tvf_return_samples=samples,
                tvf_return_n_step=n_step,
                **COMMON_ARGS
            )
    for n_step in [10, 20, 40, 80]:
        for samples in [16]:
            add_run(
                run_name=f"n_step={n_step} samples={samples}",
                tvf_return_samples=samples,
                tvf_return_n_step=n_step,
                **COMMON_ARGS
            )

def value_heads(priority: int = 0):

    COMMON_ARGS = {
        'seeds': 1,
        'subset': ATARI_3_VAL,
        'priority': priority,
        'hostname': "",
        'env_args': HARD_MODE_ARGS,
        'default_args': TVF_INITIAL_ARGS,
        'experiment': "TVF_VALUEHEAD",
    }

    # check n_steps and samples
    for value_heads in [8, 32, 128]:
        add_run(

            run_name=f"value_heads={value_heads}",
            tvf_value_samples=value_heads,
            tvf_horizon_samples=value_heads,
            **COMMON_ARGS
        )

def cluster_dropout(priority: int = 0):

    IMPROVED_ARGS = {
        'experiment': "TVF_DROPOUT_2",
        'seeds': 2,
        'subset': ATARI_3_VAL,
        'priority': priority,
        'hostname': "",
        #'device': 'cuda',
        'env_args': HARD_MODE_ARGS,
        # improvements
        'use_tvf': True,
        'tvf_mode': 'fixed',
        'gae_lambda': 0.8,
        'hidden_units': 512,
        'tvf_return_n_step': 20,
        'tvf_return_samples': 32,
        'tvf_hidden_units': 0,
        'replay_size': 1 * 128 * 128,
        'distil_batch_size': 1 * 128 * 128,
        'policy_epochs': 2,
        'distil_epochs': 2,
        'tvf_coef': 0.5,
    }

    # second attempt, no hidden layer will allow heads to be more independent.

    for value_epochs in [2]:
        for tvf_horizon_dropout in [0.5, 0.9, 0.99]:
            add_run(
                run_name=f"2{value_epochs}2 dropout={tvf_horizon_dropout}",
                default_args=TVF_INITIAL_ARGS,
                value_epochs=value_epochs,
                tvf_horizon_dropout=tvf_horizon_dropout,
                **IMPROVED_ARGS
            )
        add_run(
            run_name=f"2{value_epochs}2 reference",
            default_args=TVF_INITIAL_ARGS,
            value_epochs=value_epochs,
            tvf_horizon_dropout=0,
            **IMPROVED_ARGS
        )



def setup():

    reference(25)
    horizon()
    returns()
    value_heads()

    cluster_dropout(200)