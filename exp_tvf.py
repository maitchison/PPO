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

TVF6_ARGS = TVF5_ARGS.copy()
TVF6_ARGS.update({
    'ag_sns_min_h': 100,             # important
    'tvf_return_mode': "advanced",   # seems better than advanced2
    'distil_epochs': 2,              # better than 1
    'ag_sns_delay': int(2e6),        # earlier is better
    'ag_sns_ema_horizon': int(10e6)  # this needs to change very slowly
})

TVF7_ARGS = TVF6_ARGS.copy()
TVF7_ARGS.update({
    'ag_min_h': 100,
    'tvf_return_mode': "advanced",   # seems better than advanced2
    'distil_epochs': 2,              # better than 1?
    'ag_delay': int(2e6),            # earlier is better
    'ag_sns_ema_horizon': int(10e6)  # this needs to change very slowly
})


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


def tvf5_zp(priority: int = 0):

    # another look into rediscounting with lessons learned

    COMMON_ARGS = {
        'seeds': 3,
        'priority': priority,
        'env_args': HARD_MODE_ARGS,
        'experiment': "TVF5_ZP",
        'default_args': TVF5_ARGS,
        'epochs': 30,  # 20 is enough for these two games (needs a bit more...)
        'subset': ["Zaxxon"],
    }

    add_run(
        run_name=f"red_99",
        gamma=0.99,
        tvf_gamma=0.9999,
        override_reward_normalization_gamma=0.999,  # compromise
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"tvf_99",
        gamma=0.99,
        tvf_gamma=0.99,
        **COMMON_ARGS,
    )

    for beta in [0.3, 1.0, 3.0]:
        add_run(
            run_name=f"rn_99 beta={beta}",
            gamma=0.99,
            tvf_gamma=0.9999,
            distil_order="before_policy",
            distil_period=2,
            distil_epochs=2,
            tvf_return_mode="advanced",
            override_reward_normalization_gamma=0.9999,  # no override, just renorm
            distil_renormalize=True, # this is probably not needed so long as tvf_gamma=gamma.
            distil_beta=beta,
            **COMMON_ARGS,
        )

    # alternative strategy with rediscounting and renorm
    # the idea here is our features will be better
    for beta in [0.3, 0.5, 1.0, 3.0]:
        add_run(
            run_name=f"rnd_99 beta={beta}",
            gamma=0.99,
            tvf_gamma=0.9999,
            distil_order="before_policy",
            distil_period=2,
            distil_epochs=2,
            tvf_return_mode="advanced",
            override_reward_normalization_gamma=0.9999,  # no override, just renorm
            distil_renormalize=True,  # this is probably not needed so long as tvf_gamma=gamma.
            distil_rediscount=True,
            distil_beta=beta,
            **COMMON_ARGS,
        )



def tvf5_yp(priority: int = 0):

    # really just want to know if this works or not...

    COMMON_ARGS = {
        'seeds': 3,
        'priority': priority,
        'env_args': HARD_MODE_ARGS,
        'experiment': "TVF5_YP",
        'default_args': TVF5_ARGS,
        'epochs': 30,  # 20 is enough for these two games (needs a bit more...)
        'subset': ["YarsRevenge"],
    }

    add_run(
        run_name=f"red_99",
        gamma=0.99,
        tvf_gamma=0.9999,
        override_reward_normalization_gamma=0.999,  # compromise
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"red_99 do_distil",
        gamma=0.99,
        tvf_gamma=0.9999,
        override_reward_normalization_gamma=0.999,  # compromise
        distil_epochs=0,
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"tvf_99",
        gamma=0.99,
        tvf_gamma=0.99,
        **COMMON_ARGS,
    )

    # attempts to fix...
    add_run(
        run_name=f"red_99 distil_reweighing",
        gamma=0.99,
        tvf_gamma=0.9999,
        override_reward_normalization_gamma=0.999,  # compromise,
        distil_reweighing=True,
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"red_99 distil_rediscount",
        gamma=0.99,
        tvf_gamma=0.9999,
        override_reward_normalization_gamma=0.999,  # compromise,
        distil_rediscount=True,
        **COMMON_ARGS,
    )

    for lvt in [1.0]:
        add_run(
            run_name=f"red_99 lvt={lvt}",
            gamma=0.99,
            tvf_gamma=0.9999,
            override_reward_normalization_gamma=0.999,  # compromise,
            distil_loss_value_target=lvt,
            **COMMON_ARGS,
        )

    for orng in [0.99, 0.9999]:
        add_run(
            run_name=f"red_99 orng={orng}",
            gamma=0.99,
            tvf_gamma=0.9999,
            override_reward_normalization_gamma=orng,
            **COMMON_ARGS,
        )

    # more lvt...
    # also... we now get nice distil curves
    COMMON_ARGS['experiment'] = "TVF5_YP2"
    for lvt in [0.3, 1.0, 3.0]:
        add_run(
            run_name=f"red_99 lvt_mean={lvt}",
            gamma=0.99,
            tvf_gamma=0.9999,
            override_reward_normalization_gamma=0.999,  # compromise,
            distil_loss_value_target=lvt,
            distil_lvt_mode="mean",
            **COMMON_ARGS,
        )

    COMMON_ARGS['experiment'] = "TVF5_YP3"
    add_run(
        run_name=f"red_99 distil_renormalize",
        gamma=0.99,
        tvf_gamma=0.9999,
        override_reward_normalization_gamma=0.999,  # compromise,
        distil_renormalize=True,
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"red_99 distil_renormalize lvt_mean=1.0",
        gamma=0.99,
        tvf_gamma=0.9999,
        override_reward_normalization_gamma=0.999,  # compromise,
        distil_loss_value_target=1.0,
        distil_lvt_mode="mean",
        distil_renormalize=True,
        **COMMON_ARGS,
    )

    # just to get distil loss curves
    COMMON_ARGS['experiment'] = "TVF5_YP4"
    COMMON_ARGS['seeds'] = 1

    add_run(
        run_name=f"tvf_99 [ref]",
        gamma=0.99,
        tvf_gamma=0.99,
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"tvf_99 beta=0.1",
        gamma=0.99,
        tvf_gamma=0.99,
        distil_beta=0.1,
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"tvf_99 beta=10",
        gamma=0.99,
        tvf_gamma=0.99,
        distil_beta=10,
        ** COMMON_ARGS,
    )

    # this run will be useful, the renormalization factor should be close to 1...
    add_run(
        run_name=f"tvf_99 renormalization",
        gamma=0.99,
        tvf_gamma=0.99,
        distil_renormalize=True,
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"red_99 renormalization",
        gamma=0.99,
        tvf_gamma=0.9999,
        override_reward_normalization_gamma=0.999,  # compromise
        distil_renormalize=True,
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"red_99 [ref]",
        gamma=0.99,
        tvf_gamma=0.9999,
        override_reward_normalization_gamma=0.999,  # compromise
        **COMMON_ARGS,
    )

def tvf7_first(priority: int = 0):

    COMMON_ARGS = {
        'seeds': 2,
        'priority': priority,
        'env_args': HARD_MODE_ARGS,
        'experiment': "TVF7",
        'default_args': TVF6_ARGS,
        'epochs': 50,  # 20 is enough for these two games
        'subset': ATARI_3_VAL + ATARI_5 + ["Skiing", "CrazyClimber", "Breakout", "BeamRider"],

        # put these on the cluster so they actually get done...
        'hostname': "cluster",
        'device': 'cuda',
    }

    # first hit

    add_run(
        run_name=f"tvf",
        gamma=0.9999,
        tvf_gamma=0.9999,
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"auto_best",
        tvf_gamma=0.9999,
        use_ag=True,
        ag_mode="h_best",
        ag_min_h=30,
        ag_max_h=10000,
        debug_log_rediscount_curve=True,
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"auto_sns",
        tvf_gamma=0.9999,
        use_ag=True,
        ag_mode="sns",
        ag_min_h=100,
        ag_max_h=10000,
        debug_log_rediscount_curve=True,
        **COMMON_ARGS,
    )


def tvf6_curve(priority: int = 0):

    COMMON_ARGS = {
        'seeds': 1,
        'priority': priority,
        'env_args': HARD_MODE_ARGS,
        'experiment': "TVF6_AUTO3",
        'default_args': TVF6_ARGS,
        'epochs': 50,  # 20 is enough for these two games
        'subset': ATARI_3_VAL + ["Skiing", "CrazyClimber", "Breakout"],

        # put these on the cluster so they actually get done...
        # 'hostname': "cluster",
        # 'device': 'cuda',
    }

    # use distil curve for auto adjustment..
    add_run(
        run_name=f"auto4",
        tvf_gamma=0.9999,
        use_ag=True,
        ag_mode="h_best",
        ag_sns_min_h=30,
        ag_sns_max_h=10000,
        debug_log_rediscount_curve=True,
        **COMMON_ARGS,
    )

def tvf6_auto(priority: int = 0):

    COMMON_ARGS = {
        'seeds': 3,
        'priority': priority,
        'env_args': HARD_MODE_ARGS,
        'experiment': "TVF6_AUTO",
        'default_args': TVF6_ARGS,
        'epochs': 50,  # 20 is enough for these two games
        'subset': ATARI_3_VAL,

         # put these on the cluster so they actually get done...
         # 'hostname': "cluster",
         # 'device': 'cuda',
    }

    # add_run(
    #     run_name=f"tvf_9999",
    #     gamma=0.9999,
    #     tvf_gamma=0.9999,
    #     **COMMON_ARGS,
    # )
    #
    # add_run(
    #     run_name=f"tvf_999",
    #     gamma=0.999,
    #     tvf_gamma=0.999,
    #     **COMMON_ARGS,
    # )
    #
    # add_run(
    #     run_name=f"tvf_99",
    #     gamma=0.99,
    #     tvf_gamma=0.99,
    #     **COMMON_ARGS,
    # )
    #
    # add_run(
    #     run_name=f"red_99",
    #     gamma=0.99,
    #     tvf_gamma=0.9999,
    #     distil_rediscount=True,
    #     distil_renormalize=True,
    #     **COMMON_ARGS,
    # )

    add_run(
        run_name=f"auto1",
        tvf_gamma=0.9999,
        distil_rediscount=True,
        distil_renormalize=True,
        use_ag=True,
        **COMMON_ARGS,
    )

    add_run(
        run_name=f"auto2",
        tvf_gamma=0.9999,
        use_ag=True,
        **COMMON_ARGS,
    )

    COMMON_ARGS['experiment'] = "TVF6_AUTO2"
    COMMON_ARGS['seeds'] = 1

    # allow scaling up and down... interesting idea...
    add_run(
        run_name=f"auto3",
        tvf_gamma=0.999,
        debug_log_rediscount_curve=True,
        use_ag=True,
        **COMMON_ARGS,
    )
    COMMON_ARGS['subset'] = ATARI_3_VAL + ["Skiing", "CrazyClimber", "Breakout", "Centipede"]

    # really just adding som extra envs and more distil curve tracking
    add_run(
        run_name=f"auto_redo",
        tvf_gamma=0.9999, # just want to get debug plots...
        debug_log_rediscount_curve=True,
        use_ag=True,
        **COMMON_ARGS,
    )

    # add_run(
    #     run_name=f"auto1",
    #     gamma=0.99,
    #     tvf_gamma=0.9999,
    #     distil_rediscount=True,
    #     distil_renormalize=True,
    #     use_ag=True,
    #     **COMMON_ARGS,
    # )


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
    for tvl in [0.3, 1.0]:
        add_run(
            run_name=f"tvf {tvl}",
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
    add_run(
        run_name=f"distil 2_every_1",
        distil_period=1,
        **COMMON_ARGS,
    )
    add_run( # was labeled as 1_every_2
        run_name=f"distil 2_every_2",
        distil_epochs=2,
        **COMMON_ARGS,
    )

    # todo: ref are actually tvfl, check and confirm
    # 1_every_2 is actually 2_every_2

    # check if our estimator is causing problems (advanced seems like it might be good?)
    for mode in ["exponential", "advanced", "fixed"]:
        add_run(
            run_name=f"tvf_return_mode={mode}",
            tvf_return_mode=mode,
            **COMMON_ARGS,
        )




def setup():


    tvf5_tuning()
    # tvf5_yp()
    tvf5_zp()

    tvf6_auto()
    tvf6_curve(100)
    tvf7_first()
    pass
