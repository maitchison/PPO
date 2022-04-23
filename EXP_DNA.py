from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import TVF_reference_args, DNA_reference_args
import numpy as np

ROLLOUT_SIZE = 128*128
ATARI_3_VAL_v1 = ['NameThisGame', 'WizardOfWor', 'YarsRevenge']
ATARI_5_v1 = ['Asterix', 'BattleZone', 'DoubleDunk', 'Phoenix', 'RiverRaid']

# new set...
ATARI_3_VAL = ['Assault', 'MsPacman', 'YarsRevenge']
ATARI_5_VAL = ['Assault', 'MsPacman', 'YarsRevenge', 'BankHeist', 'VideoPinball']

ATARI_5 = ['BattleZone', 'DoubleDunk', 'NameThisGame', 'Phoenix', 'Qbert']

# proposed changes:
# repeat action penality = 0.25

# These are the best settings from the HPS, but not from the axis search performed later.
DNA_HARD_ARGS_HPS = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'architecture': 'dual',
    'export_video': False,
    'epochs': 50,
    'use_compression': True,
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
    'gamma': 0.999, # was 0.997

    # other
    'observation_normalization': True, # pong (and others maybe) do not work without this, so jsut default it to on..

}

# used in the PPO Paper
PPO_ORIG_ARGS = DNA_HARD_ARGS_HPS.copy()
PPO_ORIG_ARGS.update({
    'n_steps': 128,            # no change
    'agents': 8,
    'ppo_epsilon': 0.1,        # no anneal.
    'entropy_bonus': 1e-2,     # no change
    'gamma': 0.99,
    'policy_epochs': 3,
    'td_lambda': 0.95,
    'gae_lambda': 0.95,
    'policy_mini_batch_size': 256,
    'value_epochs': 0,
    'distil_epochs': 0,
    'architecture': 'single',
})

def dna_hps(priority: int = 0):

    # second HPS
    # more distil searching
    # less n_steps more agents
    # less entropy
    # only 32 samples

    search_params = {
        # ppo params
        'entropy_bonus':     Categorical(3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
        'agents':            Categorical(64, 128, 256, 512),
        'n_steps':           Categorical(32, 64, 128),
        'gamma':             Categorical(0.99, 0.997, 0.999),
        'gae_lambda':        Categorical(0.9, 0.95, 0.975),
        # dna params
        'policy_lr':         Categorical(1e-4, 2.5e-4, 5e-4),
        'distil_lr':         Categorical(1e-4, 2.5e-4, 5e-4),
        'value_lr':          Categorical(1e-4, 2.5e-4, 5e-4),
        'td_lambda':         Categorical(0.9, 0.95, 0.975),
        'policy_epochs':     Categorical(1, 2, 3),
        'value_epochs':      Categorical(1, 2, 3),
        'distil_epochs':     Categorical(1, 2, 3),
        'distil_beta':       Categorical(0.5, 1.0, 2.0),
        'policy_mini_batch_size': Categorical(256, 512, 1024, 2048),
        'value_mini_batch_size': Categorical(256, 512, 1024, 2048),
        'distil_mini_batch_size': Categorical(256, 512, 1024, 2048),
        'replay_size':       Categorical(*[x * (8*1024) for x in [1, 2, 4, 8]]),
        'repeated_action_penalty': Categorical(0, 0.25, 1.0),
        'entropy_scaling':   Categorical(True, False),

        # replay params
        'replay_mode':       Categorical("overwrite", "sequential", "uniform", "off"),
    }

    main_params = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'architecture': 'dual',
        'export_video': False,
        'epochs': 50,
        'use_compression': True,
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
        'ppo_epsilon': 0.1,
        'hidden_units': 256,

        # tvf params
        'use_tvf': False,

        # distil / replay buffer (This would have been called h11 before
        'distil_period': 1,
        'replay_size': 0,       # off for now...
        'distil_beta': 2.0,     # was 1.0

        'replay_mode': "uniform",

        # horizon
        'gamma': 0.999, # was 0.997

        # other
        'observation_normalization': True,
    }


    def fixup_params(params):

        rollout_size = params['agents'] * params['n_steps']

        # default to using rollout size for distil
        params['distil_batch_size'] = rollout_size

        # set replay_size to 0 if it is not being used
        if params['replay_mode'] == "off":
            params["replay_size"] = 0

        # limit epochs to 6 (otherwise they will be too slow...)
        epoch_params = ['policy_epochs', 'value_epochs', 'distil_epochs']
        while sum(params[x] for x in epoch_params) > 6:
            dice_roll = np.random.randint(0, 3)
            if params[epoch_params[dice_roll]] > 1:
                params[epoch_params[dice_roll]] -= 1

        params["use_compression"] = params['replay_size'] + rollout_size > 32*1024
        params["disable_ev"] = True

    extra_seeds = {
    }

    random_search(
        "DNA_SEARCH",
        main_params,
        search_params,
        count=32,
        process_up_to=32,
        envs=ATARI_3_VAL,
        hook=fixup_params,
        priority=priority,
        run_seed_lookup=extra_seeds,
    )


# def dna_replay(priority=0):
#     HOST = ''
#     ROLLOUT_SIZE = 16*1024
#
#     for seed in [1]: # this may need 3 seeds (if results are close)
#         for env in ATARI_3_VAL:
#             args = {
#                 'env_name': env,
#                 'hostname': HOST,
#                 'priority': priority + (0 if seed == 1 else -50) - 500,
#                 'seed': seed,
#                 'use_compression': True,
#             }
#             for replay_size in [1, 4, 16, 64]:
#                 args['priority'] = priority
#                 add_job(
#                     "DNA_REPLAY",
#                     run_name=f"game={env} uniform {replay_size}x ({seed})",
#                     replay_size=replay_size * ROLLOUT_SIZE,
#                     replay_mode="uniform",
#                     **args,
#                     default_params=DNA_HARD_ARGS,
#                 )
#                 args['priority'] = priority - 100
#                 add_job(
#                     "DNA_REPLAY",
#                     run_name=f"game={env} mixed {replay_size}x ({seed})",
#                     replay_size=replay_size * ROLLOUT_SIZE,
#                     replay_mode="uniform",
#                     replay_mixing=True,
#                     **args,
#                     default_params=DNA_HARD_ARGS,
#                 )
#
#                 add_job(
#                     "DNA_REPLAY",
#                     run_name=f"game={env} sequential {replay_size}x ({seed})",
#                     replay_size=replay_size * ROLLOUT_SIZE,
#                     replay_mode="sequential",
#                     **args,
#                     default_params=DNA_HARD_ARGS,
#                 )
#                 add_job(
#                     "DNA_REPLAY",
#                     run_name=f"game={env} replay=ppg ({seed})",
#                     replay_size=replay_size * ROLLOUT_SIZE,
#                     replay_mode="sequential",
#                     distil_period=replay_size,
#                     distil_batch_size=replay_size * ROLLOUT_SIZE,  # huge batch... but only every x updates
#                     **args,
#                     default_params=DNA_HARD_ARGS,
#                 )
#
#             args['priority'] = priority + 10
#
#             add_job(
#                 "DNA_REPLAY",
#                 run_name=f"game={env} replay=off ({seed})",
#                 replay_size=0 * ROLLOUT_SIZE,
#                 replay_mode="sequential",
#                 **args,
#                 default_params=DNA_HARD_ARGS,
#             )


def dna_gae(priority=0):
    """
    Demonstrate that GAE for advantages and for return values can be different.
    """

    # would be nice to check noise levels too..., but maybe do that later?

    HOST = ''
    for seed in [1, 2, 3, 5]: # this will need 3 seeds (results are close)
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority - (seed - 1) * 55,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': seed != 3, # need to know this on one seed at least
                'distil_beta': 1.0,
                'abs_mode': 'off',
            }
            for gae_lambda in [0.6, 0.8, 0.9, 0.95, 0.975]: # guessing GAE wants lower...
                for td_lambda in [0.8, 0.95]:
                    add_job(
                        "DNA_GAE",
                        run_name=f"game={env} td_lambda={td_lambda} gae_lambda={gae_lambda} ({seed})",
                        gae_lambda=gae_lambda,
                        td_lambda=td_lambda,
                        policy_epochs=2,
                        value_epochs=1,
                        distil_epochs=2,
                        **args,
                        default_params=DNA_HARD_ARGS_HPS,
                    )
                for td_lambda in [0.95]:
                    args['priority'] = priority - (seed - 1) * 55 - 500
                    add_job(
                        "PPO_GAE",
                        run_name=f"game={env} td_lambda={td_lambda} gae_lambda={gae_lambda} ({seed})",
                        gae_lambda=gae_lambda,
                        td_lambda=td_lambda,
                        policy_epochs=1,
                        value_epochs=0,
                        distil_epochs=0,
                        architecture='single',
                        **args,
                        default_params=DNA_HARD_ARGS_HPS,
                    )

    # noise level results...
    for seed in [4]: # this will need 3 seeds (results are close)
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': '',
                'priority': priority - (seed - 1) * 55,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': False,
                'distil_beta': 1.0,
                'abs_mode': 'shadow',
            }
            for gae_lambda in [0.6, 0.8, 0.9, 0.95, 0.975]:  # guessing GAE wants lower...
                for td_lambda in [0.8, 0.95]:
                    add_job(
                        "DNA_GAE",
                        run_name=f"game={env} td_lambda={td_lambda} gae_lambda={gae_lambda} ({seed})",
                        gae_lambda=gae_lambda,
                        td_lambda=td_lambda,
                        policy_epochs=2,
                        value_epochs=1,
                        distil_epochs=2,
                        **args,
                        default_params=DNA_HARD_ARGS_HPS,
                    )


def dna_are(priority=0):
    """
    ...
    """

    # would be nice to check noise levels too..., but maybe do that later?

    HOST = ''
    for seed in [1]:
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority - (seed - 1) * 55,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': False, # need to know this on one seed at least
                'distil_beta': 1.0,
                'are_mode': 'policy',
            }
            for target in [100, 300]:
                add_job(
                    "DNA_ARE5",
                    run_name=f"game={env} target_a={target} ({seed})",
                    are_target_p=target,
                    policy_epochs=2,
                    value_epochs=1,
                    distil_epochs=2,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
            args['are_mode'] = 'shadow'
            add_job(
                "DNA_ARE5",
                run_name=f"game={env} shadow ({seed})",
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

    return


    HOST = ''
    for seed in [1]:
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority - (seed - 1) * 55,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': False, # need to know this on one seed at least
                'distil_beta': 1.0,
                'are_mode': 'on',
            }
            # add_job(
            #     "DNA_ARE",
            #     run_name=f"game={env} target_v=10 target_a=100 ({seed})",
            #     are_target_a=100,
            #     are_target_v=10,
            #     policy_epochs=2,
            #     value_epochs=1,
            #     distil_epochs=2,
            #     **args,
            #     default_params=DNA_HARD_ARGS_HPS,
            # )
            add_job(
                "DNA_ARE2",
                run_name=f"game={env} target_v=15 target_a=175 ({seed})",
                are_target_a=175,
                are_target_v=15,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            args['are_mode'] = 'shadow'
            add_job(
                "DNA_ARE",
                run_name=f"game={env} off ({seed})",
                are_target_a=150,
                are_target_v=15,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

def merge_dict(a:dict, b:dict):
    c = a.copy()
    for k, v in b.items():
        c[k] = v
    return c


def dna_final(priority=0):
    """
    Our main results...
    (including ablations)
    """

    HOST = '' # ML

    for seed in [1]:
        for env in ATARI_5:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority - (seed - 1) * 55,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': False,
                'distil_beta': 1.0,

                'terminal_on_loss_of_life': True,
                'reward_clipping': 1,
                'full_action_space': False,
                'repeat_action_probability': 0.0,
            }

            # easy mode run...
            add_job(
                "DNA_EASY",
                run_name=f"game={env} dna_tuned ({seed})",  # dna
                gae_lambda=0.8,
                td_lambda=0.95,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            add_job(
                "DNA_EASY",
                run_name=f"game={env} ppo_tuned_fat ({seed})",
                gae_lambda=0.95,
                td_lambda=0.95,
                policy_epochs=1,
                value_epochs=0,
                distil_epochs=0,
                architecture='single',
                network='nature_fat',
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

    # suplementroy results, 1 seed

    # impala is a bonus for later... (3x slower)

    # add_job(
    #     "DNA_FINAL",
    #     run_name=f"game={env} dna_impala ({seed})",
    #     gae_lambda=0.8,
    #     td_lambda=0.95,
    #     policy_epochs=2,
    #     value_epochs=1,
    #     distil_epochs=2,
    #     hidden_units=256,
    #     network='impala',
    #     **merge_dict(args, {'hostname': "desktop", 'priority': -100}), # faster on desktop
    #     default_params=DNA_HARD_ARGS_HPS,
    # )
    #
    # add_job(
    #     "DNA_FINAL2",
    #     run_name=f"game={env} dna_impala_fast ({seed})",  # far less parameters..
    #     gae_lambda=0.8,
    #     td_lambda=0.95,
    #     policy_epochs=2,
    #     value_epochs=1,
    #     distil_epochs=2,
    #     hidden_units=256,
    #     network='impala_fast',
    #     **merge_dict(args, {'hostname': "desktop", 'priority': -100}),  # faster on desktop
    #     default_params=DNA_HARD_ARGS_HPS,
    # )

    # add_job(
    #     "DNA_FINAL",
    #     run_name=f"game={env} dna_fat ({seed})", # used in agent_57
    #     gae_lambda=0.8,
    #     td_lambda=0.95,
    #     policy_epochs=2,
    #     value_epochs=1,
    #     distil_epochs=2,
    #     hidden_units=512, # default
    #     network='nature_fat',
    #     **args,
    #     default_params=DNA_HARD_ARGS_HPS,
    # )
    #
    #

    # add_job(
    #     "DNA_FINAL",
    #     run_name=f"game={env} dna ({seed})", # dna-no_tune
    #     policy_epochs=2,
    #     value_epochs=1,
    #     distil_epochs=2,
    #     **args,
    #     default_params=DNA_HARD_ARGS_HPS,
    # )
    #
    # # todo: add no distil, but tuned
    #
    # add_job(
    #     "DNA_FINAL",
    #     run_name=f"game={env} dna_no_distil ({seed})",
    #     policy_epochs=2,
    #     value_epochs=1,
    #     distil_epochs=0,
    #     **args,
    #     default_params=DNA_HARD_ARGS_HPS,
    # )
    #
    # add_job(
    #     "DNA_FINAL",
    #     run_name=f"game={env} ppo ({seed})",
    #     policy_epochs=4,
    #     value_epochs=0,
    #     distil_epochs=0,
    #     architecture='single',
    #     network='nature',
    #     **args,
    #     default_params=DNA_HARD_ARGS_HPS,
    # )
    #
    # add_job(
    #     "DNA_FINAL",
    #     run_name=f"game={env} ppo_2 ({seed})",
    #     policy_epochs=2,
    #     value_epochs=0,
    #     distil_epochs=0,
    #     architecture='single',
    #     network='nature',
    #     **args,
    #     default_params=DNA_HARD_ARGS_HPS,
    # )
    #
    # # add_job(
    # #     "DNA_FINAL",
    # #     run_name=f"game={env} ppo_3 ({seed})",
    # #     policy_epochs=3,
    # #     value_epochs=0,
    # #     distil_epochs=0,
    # #     architecture='single',
    # #     network='nature',
    # #     **args,
    # #     default_params=DNA_HARD_ARGS_HPS,
    # # )
    #
    # add_job(
    #     "DNA_FINAL",
    #     run_name=f"game={env} ppo_2_fat ({seed})",
    #     policy_epochs=2,
    #     value_epochs=0,
    #     distil_epochs=0,
    #     architecture='single',
    #     network='nature_fat',
    #     **args,
    #     default_params=DNA_HARD_ARGS_HPS,
    # )
    #

    #
    # add_job(
    #     "DNA_FINAL",
    #     run_name=f"game={env} ppo_orig ({seed})",
    #     n_steps=128,            # no change
    #     agents=8,
    #     ppo_epsilon=0.1,        # no anneal.
    #     entropy_bonus=1e-2,     # no change
    #     gamma=0.99,
    #     policy_epochs=3,
    #     td_lambda=0.95,
    #     gae_lambda=0.95,
    #     policy_mini_batch_size=256,
    #
    #     value_epochs=0,
    #     distil_epochs=0,
    #     architecture='single',
    #     network='nature',
    #     **args,
    #     default_params=DNA_HARD_ARGS_HPS,
    # )
    #
    # add_job(
    #     "DNA_FINAL",
    #     run_name=f"game={env} ppo_tuned ({seed})",
    #     gae_lambda=0.8,
    #     td_lambda=0.95,
    #     policy_epochs=4,
    #     value_epochs=0,
    #     distil_epochs=0,
    #     architecture='single',
    #     network='nature',
    #     **args,
    #     default_params=DNA_HARD_ARGS_HPS,
    # )
    #
    # add_job(
    #     "DNA_FINAL",
    #     run_name=f"game={env} ppo_fat ({seed})",
    #     policy_epochs=4,
    #     value_epochs=0,
    #     distil_epochs=0,
    #     architecture='single',
    #     network='nature_fat',
    #     **args,
    #     default_params=DNA_HARD_ARGS_HPS,
    # )

    # main result, 3 seeds
    for seed in [1, 2, 3]:
        for env in ATARI_5:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority - (seed - 1) * 55,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': False,
                'distil_beta': 1.0,
            }

            add_job(
                "DNA_FINAL",
                run_name=f"game={env} ppo_tuned_fat ({seed})",
                gae_lambda=0.95,
                td_lambda=0.95,
                policy_epochs=1,
                value_epochs=0,
                distil_epochs=0,
                architecture='single',
                network='nature_fat',
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

            add_job(
                "DNA_FINAL",
                run_name=f"game={env} dna_tuned ({seed})",  # dna
                gae_lambda=0.8,
                td_lambda=0.95,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )


def dna_beta(priority=0):
    HOST = ''
    for seed in [1, 2, 3]:
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority - (seed-1) * 10,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,
                'disable_ev': True,
            }
            for beta in [0.5, 1.0, 2.0]:
                args['priority'] = priority - (seed-1) * 10 - int(beta)
                add_job(
                    "DNA_BETA",
                    run_name=f"game={env} beta={beta} ({seed})",
                    distil_beta=beta,
                    policy_epochs=2,
                    value_epochs=1,
                    distil_epochs=2,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
    for seed in [1]: # three seeds is needed for an accurate results (but just do one for now...)
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority - (seed-1) * 10,
                'seed': seed,
                'disable_ev': seed == 3, # only ev on seed 3...
            }
            for value_epochs in [1, 2, 3, 4]:
                if value_epochs == 0:
                    if seed != 1:
                        continue
                add_job(
                    "DNA_VALUE_EPOCHS",
                    run_name=f"game={env} value_epochs={value_epochs} ({seed})",
                    value_epochs=value_epochs,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )


def dna_mbs(priority=0):
    HOST = ''
    for seed in [1]:
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority - (seed - 1) * 10,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,
                'disable_ev': True,
            }
            for policy_epochs in [4]:
                for mbs in [8]:
                    args['priority'] = priority - (seed - 1) * 10
                    add_job(
                        "DNA_MBS",
                        run_name=f"game={env} epochs={policy_epochs} mbs={mbs}k ({seed})",
                        distil_beta=1.0,
                        policy_mini_batch_size=mbs*1024,
                        policy_epochs=policy_epochs,
                        value_epochs=1,
                        distil_epochs=2,
                        **args,
                        default_params=DNA_HARD_ARGS_HPS,
                    )

def dna_epochs(priority=0):

    # these value epochs where done early, but should probably be redone (with new codebase, and with beta=0.5)
    # HOST = ''
    # for seed in [1, 2, 3]: # three seeds is needed for an accurate results (but just do one for now...)
    #     for env in ATARI_3_VAL:
    #         args = {
    #             'env_name': env,
    #             'hostname': HOST,
    #             'priority': priority - (seed-1) * 10,
    #             'seed': seed,
    #             'disable_ev': seed == 3, # only ev on seed 3...
    #         }
    #         for value_epochs in [1, 2, 3, 4]:
    #             if value_epochs == 0:
    #                 if seed != 1:
    #                     continue
    #             add_job(
    #                 "DNA_EPOCHS",
    #                 run_name=f"game={env} value_epochs={value_epochs} ({seed})",
    #                 value_epochs=value_epochs,
    #                 **args,
    #                 default_params=DNA_HARD_ARGS_HPS,
    #             )

    # one value epoch was, indeed, best...
    PATH = f"DNA_EPOCHS"
    for seed in [1, 2, 3]:  # three seeds is needed for an accurate results
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': '',
                'priority': priority - (seed - 1) * 10,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True, # much faster
                'disable_ev': True,
                'distil_beta': 1.0,
            }
            for value_epochs in [1, 2, 3, 4]:
                args['priority'] = priority - (seed - 1) * 10 - value_epochs - 200
                distil_epochs = 2
                policy_epochs = 2
                add_job(
                    PATH,
                    run_name=f"game={env} epochs={policy_epochs}{value_epochs}{distil_epochs} ({seed})",
                    policy_epochs=policy_epochs,
                    value_epochs=value_epochs,
                    distil_epochs=distil_epochs,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
            for policy_epochs in [1, 2, 3, 4]:
                args['priority'] = priority - (seed - 1) * 10 + policy_epochs - 100
                distil_epochs = 2
                add_job(
                    PATH,
                    run_name=f"game={env} epochs={policy_epochs}{1}{distil_epochs} ({seed})",
                    policy_epochs=policy_epochs,
                    value_epochs=1,
                    distil_epochs=distil_epochs,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
            for distil_epochs in [0, 1, 2, 3, 4]:
                args['priority'] = priority - (seed - 1) * 10 + distil_epochs
                policy_epochs = 2
                add_job(
                    PATH,
                    run_name=f"game={env} epochs={policy_epochs}{1}{distil_epochs} ({seed})",
                    policy_epochs=policy_epochs,
                    value_epochs=1,
                    distil_epochs=distil_epochs,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )


def dna_mode(priority=0):
    for seed in [1]:
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': '',
                'priority': priority - (seed - 1) * 10,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True, # much faster
                'disable_ev': True,
                'distil_beta': 1.0,
            }
            for distil_mode in ['off', 'value', 'features', 'projection']:
                add_job(
                    "DNA_MODE",
                    run_name=f"game={env} distil_mode={distil_mode} ({seed})",
                    policy_epochs=2,
                    value_epochs=1,
                    distil_epochs=0 if distil_mode == "off" else 2,
                    distil_mode=distil_mode,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )


def dna_band(priority=0):
    for seed in [1]: # check if consistant accross seeds...?
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': 'desktop', # ML rig has trouble...? maybe due to cdist?
                'priority': priority - (seed - 1) * 10,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True, # much faster
                'disable_ev': True,
                'distil_beta': 1.0,
            }
            # add_job(
            #     "DNA_BAND",
            #     run_name=f"game={env} ({seed})",
            #     policy_epochs=2,
            #     value_epochs=1,
            #     distil_epochs=2,
            #     debug_log_bandpass=True,
            #     **args,
            #     default_params=DNA_HARD_ARGS_HPS,
            # )
            # better sampling...
            add_job(
                "DNA_BAND_3",
                run_name=f"game={env} norm ({seed})",
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                log_frequency_response=True,
                lfr_samples=256,
                lfr_normalize=True,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            add_job(
                "DNA_BAND_3",
                run_name=f"game={env} no_norm ({seed})",
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                log_frequency_response=True,
                lfr_samples=256,
                lfr_normalize=False,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

def dna_noise(priority=0):
    HOST = ''
    for seed in [1]:
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority,
                'seed': seed,
                'disable_ev': True,
            }
            add_job(
                "DNA_NOISE",
                run_name=f"game={env} ({seed})",
                abs_mode='shadow',
                gae_lambda=0.95,
                td_lambda=0.95, # keep it standard
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

def entropy_test2(priority=0):
    """ trying to see if UAC looks right. """
    for seed in [1]:
        for env in ['Assault']:
            args = {
                'env_name': env,
                'hostname': 'desktop',
                'priority': priority - (seed - 1) * 10 + 10,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': True,
                'distil_beta': 1.0,
            }
            add_job(
                "ENTROPY_BRB2",
                run_name=f"game={env} uac eb=0.01 ({seed})",
                use_uac=True,
                big_red_button=True,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            add_job(
                "ENTROPY_BRB2",
                run_name=f"game={env} reference ({seed})",
                use_uac=False,
                big_red_button=True,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

def entropy_test3(priority=0):

    """ see if we can get a better read on risky actions. """

    for seed in [1]:
        for env in ['Assault']:
            args = {
                'env_name': env,
                'hostname': 'desktop',
                'priority': priority - (seed - 1) * 10 + 10,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': True,
                'distil_beta': 1.0,
                'policy_epochs': 2,
                'value_epochs': 1,
                'distil_epochs': 2,
                'big_red_button': True,
            }
            for entropy_bonus in [0.01]:
                add_job(
                    "ENTROPY_BRB3",
                    run_name=f"game={env} uac eb={entropy_bonus} ({seed})",
                    use_uac=True,
                    entropy_bonus=entropy_bonus,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
                add_job(
                    "ENTROPY_BRB3",
                    run_name=f"game={env} reference eb={entropy_bonus} ({seed})",
                    use_uac=False,
                    entropy_bonus=entropy_bonus,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )


def entropy_test4(priority=0):

    """ see if we can get a better read on risky actions. """

    # goal: get prob of termination above 1% on reference with default entropy.

    for seed in [1]:
        for env in ['Assault']:
            args = {
                'env_name': env,
                'hostname': 'desktop',
                'priority': priority - (seed - 1) * 10 + 10,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': True,
                'distil_beta': 1.0,
                'policy_epochs': 2,
                'value_epochs': 1,
                'distil_epochs': 2,
            }
            for prob in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]:
                add_job(
                    "ENTROPY_BRB4",
                    run_name=f"game={env} brb_prob={prob} ({seed})",
                    use_uac=False,
                    big_red_button_prob=prob,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
            for uac in [True, False]:
                prob = -0.01
                add_job(
                    "ENTROPY_BRB4b",
                    run_name=f"game={env} brb_prob={abs(prob)} uac={uac} [hard] ({seed})",
                    use_uac=uac,
                    big_red_button_prob=prob,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
            for uac in [True, False]:
                prob = -0.01
                add_job(
                    "ENTROPY_BRB4b",
                    run_name=f"game={env} brb_prob={abs(prob)} uac={uac} [hard] ({seed})",
                    use_uac=uac,
                    big_red_button_prob=prob,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
    for seed in [1]:
        for env in ['Assault', 'Pong']:
            args['epochs'] = 30
            args['env_name'] = env
            for uac in [True, False]:
                prob = -0.01
                add_job(
                    "ENTROPY_BRB5",
                    run_name=f"game={env} brb_prob={abs(prob)} uac={uac} [hard] ({seed})",
                    use_uac=uac,
                    big_red_button_prob=prob,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )

def entropy_test6(priority=0):
    """
    Check on validation set...
    """

    for seed in [1]:
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': 'desktop',
                'priority': priority - (seed - 1) * 10 + 10,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': True,
                'distil_beta': 1.0,
                'policy_epochs': 2,
                'value_epochs': 1,
                'distil_epochs': 2,
                'gae_lambda': 0.8,
                'td_lambda': 0.95,
            }
            for eb_clip in [0.03, 0.01, 0.003]:
                add_job(
                    "ENTROPY_1",
                    run_name=f"game={env} eb={0.01} clip={eb_clip} ({seed})",
                    use_uac=True,
                    eb_clip=eb_clip,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
            eb_clip = 0.01
            eb = 0.01
            add_job(
                "ENTROPY_1",
                run_name=f"game={env} eb={eb} clip={eb_clip} no_auc ({seed})",
                use_uac=False,
                eb_clip=eb_clip,
                entropy_bonus=eb,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            # high (but safe?) entropy run
            # eb_clip = 0.01
            # eb = 0.03
            # add_job(
            #     "ENTROPY_1",
            #     run_name=f"game={env} eb={eb} clip={eb_clip} ({seed})",
            #     use_uac=True,
            #     eb_clip=eb_clip,
            #     entropy_bonus=eb,
            #     **args,
            #     default_params=DNA_HARD_ARGS_HPS,
            # )



def entropy_test(priority=0):

    # just is just a test...


    # uac_cost should stop entropy increasing, converging to a lower point
    # eb_clip should also stop entropy increasing.
    # ths makes me thing we want something that increases intropy where it can, so I"ll bump up the entropy I guess?


    HOST = ''
    for seed in [1]:
        for env in ['Assault']:
            args = {
                'env_name': env,
                'hostname': 'desktop',  # ML rig has trouble...? maybe due to cdist?
                'priority': priority - (seed - 1) * 10 + 10,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': True,
                'distil_beta': 1.0,
            }
            add_job(
                "ENTROPY_BRB",
                run_name=f"game={env} uac cost=10 eb_clip=0.01 brr ({seed})",
                use_uac=True,
                eb_cost_alpha=10.0,
                big_red_button=True,
                eb_clip=0.01,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            add_job(
                "ENTROPY_BRB",
                run_name=f"game={env} uac cost=50 eb_clip=0.01 brr ({seed})",
                use_uac=True,
                eb_cost_alpha=50.0,
                big_red_button=True,
                eb_clip=0.01,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            add_job(
                "ENTROPY_BRB",
                run_name=f"game={env} reference brr ({seed})",
                big_red_button=True,
                use_uac=False,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

        for env in ['Assault', 'CrazyClimber']:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority,
                'seed': seed,
                'disable_ev': True,
            }
            add_job(
                "ENTROPY_TEST",
                run_name=f"game={env} uac cost=10 eb_clip=0.01 epochs=4 ({seed})",
                use_uac=True,
                eb_cost_alpha=10.0,
                eb_clip=0.01,
                policy_epochs=4,
                value_epochs=1,
                distil_epochs=2,

                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            add_job(
                "ENTROPY_TEST",
                run_name=f"game={env} uac cost=10 eb_clip=0.01 epochs=4 eb=0.02 ({seed})",
                use_uac=True,
                eb_cost_alpha=10.0,
                entropy_bonus=2e-2,
                eb_clip=0.01,
                policy_epochs=4,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            add_job(
                "ENTROPY_TEST",
                run_name=f"game={env} uac cost=50 eb_clip=0.01 epochs=4 eb=0.04 ({seed})",
                use_uac=True,
                eb_cost_alpha=50.0,
                entropy_bonus=4e-2,
                eb_clip=0.01,
                policy_epochs=4,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            add_job(
                "ENTROPY_TEST",
                run_name=f"game={env} uac cost=10 eb_clip=0.01 ({seed})",
                use_uac=True,
                eb_cost_alpha=10.0,
                eb_clip=0.01,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            add_job(
                "ENTROPY_TEST",
                run_name=f"game={env} reference ({seed})",
                use_uac=False,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )


def dna_aux(priority=0):
    # quick check to see if reward prediction as aux task helps at all...
    HOST = ''
    for seed in [1]:
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority,
                'seed': seed,
                'disable_ev': True,
            }
            for aux_epochs in [0, 1, 2]:
                add_job(
                    "DNA_AUX",
                    run_name=f"game={env} aux_epochs={aux_epochs} ({seed})",
                    replay_size=64*1024,
                    aux_epochs=aux_epochs,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )

def ppo_tuning(priority):
    # tuning for ppo
    # one value epoch was, indeed, best...
    for seed in [1, 2, 3]:
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': '',
                'priority': priority - (seed - 1) * 50,
                'seed': seed,

                'value_epochs': 0,
                'distil_epochs': 0,
                'architecture': 'single',
                'network': 'nature_fat',
            }

            add_job(
                "PPO_TUNING",
                run_name=f"game={env} reference ({seed})",
                **args,
                default_params=PPO_ORIG_ARGS,
            )

            for policy_epochs in [1, 2, 3, 4]:
                args['priority'] = priority - (seed - 1) * 10 + policy_epochs
                gae_lambda = 0.95
                add_job(
                    "PPO_EPOCHS",
                    run_name=f"game={env} epochs={policy_epochs} lambda={gae_lambda} ({seed})",
                    policy_epochs=policy_epochs,
                    gae_lambda=gae_lambda,
                    td_lambda=gae_lambda,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
            for gae_lambda in [0.8, 0.9, 0.95, 0.975]:
                policy_epochs = 1
                add_job(
                    "PPO_EPOCHS",
                    run_name=f"game={env} epochs={policy_epochs} lambda={gae_lambda} ({seed})",
                    policy_epochs=policy_epochs,
                    gae_lambda=gae_lambda,
                    td_lambda=gae_lambda,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )


def csgo(priority):

    for seed in [1]:
        for env in ["NameThisGame"]:
            args = {
                'env_name': env,
                'hostname': '',
                'priority': priority - (seed - 1) * 50,
                'seed': seed,
                'policy_epochs': 1,
                'value_epochs': 0,
                'distil_epochs': 0,
                'architecture': 'single',
                'network': 'nature',
                'epochs': 10,
            }

            add_job(
                "CSGO_1",
                run_name=f"game={env} reference ({seed})",
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

            # for lr in [3e-3, 1e-3, 1e-4]:
            #     add_job(
            #         "CSGO_1",
            #         run_name=f"game={env} csgo={lr} clip=0.01 friction=0.01 decay=0.99 ({seed})",
            #         optimizer='csgo',
            #         policy_lr=lr,
            #         **args,
            #         default_params=DNA_HARD_ARGS_HPS,
            #     )
            #
            #     add_job(
            #         "CSGO_1",
            #         run_name=f"game={env} sgd={lr} ({seed})",
            #         optimizer='sgd',
            #         policy_lr=lr,
            #         **args,
            #         default_params=DNA_HARD_ARGS_HPS,
            #     )

            # attempt 2
            args = {
                'env_name': env,
                'hostname': '',
                'priority': priority - (seed - 1) * 50,
                'seed': seed,
                'gae_lambda': 0.8,
                'td_lambda': 0.95,
                'policy_epochs': 2,
                'value_epochs': 1,
                'distil_epochs': 2,
                'epochs': 10,
            }

            add_job(
                "CSGO_2",
                run_name=f"game={env} reference ({seed})",
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

            for mgn in [1.0, 2.5, 5.0, 10.0]:
                add_job(
                    "CSGO_2",
                    run_name=f"game={env} global_norm={mgn} ({seed})",
                    max_grad_norm=mgn,
                    grad_clip_mode='global_norm',
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
            for clip in [0.1, 0.01, 0.001]:
                add_job(
                    "CSGO_2",
                    run_name=f"game={env} marcus1={clip} ({seed})",
                    max_grad_norm=clip,
                    grad_clip_mode='marcus1',
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
            for clip in [0.1, 0.01, 0.001]:
                add_job(
                    "CSGO_2",
                    run_name=f"game={env} marcus2={clip} ({seed})",
                    max_grad_norm=clip,
                    grad_clip_mode='marcus2',
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )

            for lr in [1e-2, 1e-3, 1e-4]:

                pass

                # did not work due to no RMSprop...
                # add_job(
                #     "CSGO_2",
                #     run_name=f"game={env} csgo={lr} clip=0.01 friction=0.01 decay=0.99 ({seed})",
                #     policy_optimizer='csgo',
                #     csgo_friction=0.01,
                #     csgo_decay=0.99,
                #     csgo_clip=0.01,
                #     policy_lr=lr,
                #     **args,
                #     default_params=DNA_HARD_ARGS_HPS,
                # )

                # add_job(
                #     "CSGO_2",
                #     run_name=f"game={env} sgd={lr} ({seed})",
                #     policy_optimizer='sgd',
                #     policy_lr=lr,
                #     **args,
                #     default_params=DNA_HARD_ARGS_HPS,
                # )

            ##########################
            # attempt 3
            ##########################

            # epsilon test...
            for adam_epsilon in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]: # my guess is 1e-6 (not 1e-5)
                add_job(
                    "CSGO_3",
                    run_name=f"game={env} adam_epsilon={adam_epsilon} ({seed})",
                    max_grad_norm=5.0,
                    grad_clip_mode='global_norm',
                    adam_epsilon=adam_epsilon,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )

            for lr in [2.5e-4, 1e-4, 1e-3]:
                for clip in [2.0, 3.0, 4.0, 999.0]:
                    add_job(
                        "CSGO_3",
                        run_name=f"game={env} csgo={lr} clip={clip} friction=0.01 decay=0.99 ({seed})",
                        policy_optimizer='csgo',
                        csgo_friction=0.01,
                        csgo_decay=0.99,
                        csgo_clip=clip,
                        policy_lr=lr,
                        **args,
                        default_params=DNA_HARD_ARGS_HPS,
                    )
            add_job(
                "CSGO_3",
                run_name=f"game={env} csgo={lr} clip={4.0} friction=0.1 decay=0.99 ({seed})",
                policy_optimizer='csgo',
                csgo_friction=0.1,
                csgo_decay=0.99,
                csgo_clip=4.0,
                policy_lr=lr,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

            for clip in [0.03, 0.01, 0.005, 0.003, 0.004, 0.002, 0.001, 0.0003]:
                add_job(
                    "CSGO_3",
                    run_name=f"game={env} marcus2={clip} ({seed})",
                    max_grad_norm=clip,
                    grad_clip_mode='marcus2',
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )
                add_job(
                    "CSGO_3",
                    run_name=f"game={env} marcus3={clip} ({seed})",
                    max_grad_norm=clip,
                    grad_clip_mode='marcus3',
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )

def other(priority):

    # want to check, amoung other things, batch_norm, channels, and beta1

    OTHER_HARD_ARGS_HPS = DNA_HARD_ARGS_HPS.copy()
    OTHER_HARD_ARGS_HPS['max_grad_norm'] = 5.0
    OTHER_HARD_ARGS_HPS['value_epochs'] = 1

    for seed in [1]:
        for env in ["NameThisGame"]:
            args = {
                'env_name': env,
                'hostname': '',
                'seed': seed,
                'epochs': 10,
            }

            args['priority'] =priority - (seed - 1) * 50 + 50

            add_job(
                "OTHER_1",
                run_name=f"game={env} reference ({seed})",
                **args,
                default_params=OTHER_HARD_ARGS_HPS,
            )

            args['priority'] = priority - (seed - 1) * 50 + 10

            for clip in [3, 5, 10, 20, 50]:
                add_job(
                    "CSGO_4",
                    run_name=f"game={env} marcus4={clip} epsilon=1e-6 ({seed})",
                    max_grad_norm=clip,
                    grad_clip_mode='marcus4',
                    adam_epsilon=1e-6,
                    **args,
                    default_params=OTHER_HARD_ARGS_HPS,
                )

            args['priority'] = priority - (seed - 1) * 50 + 99

            add_job(
                "CSGO_4",
                run_name=f"game={env} reference ({seed})",
                max_grad_norm=5.0,
                adam_epsilon=1e-6,
                **args,
                default_params=OTHER_HARD_ARGS_HPS,
            )

            args['priority'] = priority - (seed - 1) * 50

            for csgo_friction in [0.003, 0.01, 0.03]:
                clip = 0.003 # best from before
                add_job(
                    "OTHER_1",
                    run_name=f"game={env} marcus3={clip} friction={csgo_friction} ({seed})",
                    max_grad_norm=clip,
                    grad_clip_mode='marcus3',
                    csgo_friction=csgo_friction,
                    **args,
                    default_params=OTHER_HARD_ARGS_HPS,
                )

            for beta1 in [0, 0.8, 0.9, 0.95, 0.975]:
                if beta1 == 0.9:
                    continue
                add_job(
                    "OTHER_1",
                    run_name=f"game={env} beta1={beta1} ({seed})",
                    adam_beta1=beta1,
                    **args,
                    default_params=OTHER_HARD_ARGS_HPS,
                )

            for base_channels in [16, 32, 64]:
                if base_channels == 32:
                    continue
                add_job(
                    "OTHER_1",
                    run_name=f"game={env} policy_base_channels={base_channels} ({seed})",
                    policy_network_args="{'base_channels':"+str(base_channels)+"}",
                    **args,
                    default_params=OTHER_HARD_ARGS_HPS,
                )

            for weight_decay in [0, 0.01]:
                if weight_decay == 0:
                    continue
                add_job(
                    "OTHER_2",
                    run_name=f"game={env} weight_decay={weight_decay} ({seed})",
                    policy_weight_decay=weight_decay,
                    **args,
                    default_params=OTHER_HARD_ARGS_HPS,
                )

            for policy_norm in ['off', 'layer', 'batch']:
                for value_norm in ['off', 'layer', 'batch']:
                    if policy_norm == value_norm == "off":
                        continue
                    add_job(
                        "OTHER_1",
                        run_name=f"game={env} policy_norm={policy_norm} value_norm={value_norm} ({seed})",
                        policy_network_args="{'norm':'" + policy_norm + "'}",
                        value_network_args="{'norm':'" + value_norm + "'}",
                        **args,
                        default_params=OTHER_HARD_ARGS_HPS,
                    )

def csgo_5(priority):

    # want to check, amoung other things, batch_norm, channels, and beta1

    OTHER_HARD_ARGS_HPS = DNA_HARD_ARGS_HPS.copy()
    OTHER_HARD_ARGS_HPS['max_grad_norm'] = 5.0
    OTHER_HARD_ARGS_HPS['value_epochs'] = 1
    OTHER_HARD_ARGS_HPS['adam_beta1'] = 0.95

    for seed in [1]:
        for env in ["NameThisGame"]:
            args = {
                'env_name': env,
                'hostname': '',
                'seed': seed,
                'gae_lambda': 0.8,
                'td_lambda': 0.95,
                'policy_epochs': 2,
                'value_epochs': 1,
                'distil_epochs': 2,
                'epochs': 50, # make 50
            }

            args['priority'] =priority - (seed - 1) * 50 + 50

            add_job(
                "CSGO_5",
                run_name=f"game={env} reference ({seed})",
                **args,
                default_params=OTHER_HARD_ARGS_HPS,
            )

            args['priority'] = priority - (seed - 1) * 50 + 10

            # heavy clipping
            # for clip in [0.01, 0.001]:
            #     for beta1 in [0, 0.95]:
            #         for friction in [1.0, 0.1, 0.01]:
            #             add_job(
            #                 "CSGO_5",
            #                 run_name=f"game={env} marcus2={clip} beta1={beta1} friction={friction} ({seed})",
            #                 max_grad_norm=clip,
            #                 adam_beta1=beta1,
            #                 csgo_friction=friction,
            #                 grad_clip_mode='marcus2',
            #                 **args,
            #                 default_params=OTHER_HARD_ARGS_HPS,
            #             )

            # even more heavy clipping (and logging of redisual part)
            # interaction between clipping and momentum (with friction fixed)
            for clip in [0.001, 0.0003, 0.0001]: # add 0.01, and maybe others...
                for beta1 in [0, 0.5, 0.95]:
                    for friction in [0.01]:
                        add_job(
                            "CSGO_6",
                            run_name=f"game={env} marcus2={clip} beta1={beta1} friction={friction} ({seed})",
                            max_grad_norm=clip,
                            adam_beta1=beta1,
                            csgo_friction=friction,
                            grad_clip_mode='marcus2',
                            **args,
                            default_params=OTHER_HARD_ARGS_HPS,
                        )

            # tuning friction...
            for clip in [0.001]:
                for beta1 in [0.95]:
                    for friction in [0.03, 0.003]:
                        add_job(
                            "CSGO_6",
                            run_name=f"game={env} marcus2={clip} beta1={beta1} friction={friction} ({seed})",
                            max_grad_norm=clip,
                            adam_beta1=beta1,
                            csgo_friction=friction,
                            grad_clip_mode='marcus2',
                            **args,
                            default_params=OTHER_HARD_ARGS_HPS,
                        )

            args['priority'] = priority - (seed - 1) * 50 + 15

            for clip in [10]:
                for beta1 in [0.95]:
                    for friction in [0.01]:
                        add_job(
                            "CSGO_6",
                            run_name=f"game={env} marcus4={clip} beta1={beta1} friction={friction} ({seed})",
                            max_grad_norm=clip,
                            adam_beta1=beta1,
                            csgo_friction=friction,
                            grad_clip_mode='marcus4',
                            **args,
                            default_params=OTHER_HARD_ARGS_HPS,
                        )

            # marcus 5 mode
            # did not work at all...
            # for clip in [40]:
            #     for beta1 in [0.95]:
            #         for friction in [0.01]:
            #             add_job(
            #                 "CSGO_6",
            #                 run_name=f"game={env} marcus5={clip} beta1={beta1} friction={friction} ({seed})",
            #                 max_grad_norm=clip,
            #                 adam_beta1=beta1,
            #                 csgo_friction=friction,
            #                 grad_clip_mode='marcus5',
            #                 **args,
            #                 default_params=OTHER_HARD_ARGS_HPS,
            #             )




def entropy_scaling(priority=0):
    # theory is entropy_bonus should scale with policy epochs...
    # have a look and see if entropy is about the same?
    for seed in [1]:
        for env in ATARI_3_VAL[0:1]:
            args = {
                'env_name': env,
                'hostname': '',
                'priority': priority - (seed - 1) * 10,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True, # much faster
                'disable_ev': True,
                'distil_beta': 1.0,
                'epochs': 20, # this should be enough to see how things are going..
            }
            for policy_epochs in [4]:
                for factor in [1/4, 1/2, 1, 2, 4]:
                    # the idea here is entropy bonus should increase with epochs... see if epochs=4 works?
                    entropy_bonus = round(1e-2 * factor, 4)
                    # copy policy_epochs=1 from other...
                    add_job(
                        "ENTROPY_SCALING",
                        run_name=f"game={env} epochs={policy_epochs} entropy_bonus={entropy_bonus} ({seed})",
                        entropy_bonus=entropy_bonus,
                        policy_epochs=policy_epochs,
                        value_epochs=1,
                        distil_epochs=2,
                        **args,
                        default_params=DNA_HARD_ARGS_HPS,
                    )

def setup(priority_modifier=0):
    #dna_hps(0)     # done!
    #dna_noise(50)  # done!
    # dna_epochs(-100) # done?
    # ppo_tuning(-100)
    csgo(300)
    csgo_5(400)
    other(350)

    # dna_gae(-100) # ...
    dna_final(-100)


