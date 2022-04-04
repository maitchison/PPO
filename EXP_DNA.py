from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import TVF_reference_args, DNA_reference_args
import numpy as np

ROLLOUT_SIZE = 128*128
ATARI_3_VAL_v1 = ['NameThisGame', 'WizardOfWor', 'YarsRevenge']
ATARI_5_v1 = ['Asterix', 'BattleZone', 'DoubleDunk', 'Phoenix', 'RiverRaid']

# new set...
ATARI_3_VAL = ['Assault', 'MsPacman', 'YarsRevenge']
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

    'dna_dual_constraint': 0,

    # horizon
    'gamma': 0.999, # was 0.997

    # other
    'observation_normalization': True, # pong (and others maybe) do not work without this, so jsut default it to on..

}

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

        'dna_dual_constraint': 0,

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
    for seed in [1, 2, 3]: # this will need 3 seeds (results are close)
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
            for gae_lambda in [0.8, 0.9, 0.95]: # guessing GAE wants lower...
                for td_lambda in [0.9, 0.95, 0.975]: # and td wants higher...
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
                    # add_job(
                    #     "PPO_GAE",
                    #     run_name=f"game={env} td_lambda={td_lambda} gae_lambda={gae_lambda} ({seed})",
                    #     gae_lambda=gae_lambda,
                    #     td_lambda=td_lambda,
                    #     architecture='single',
                    #     **args,
                    #     default_params=DNA_HARD_ARGS_HPS,
                    # )


def dna_final(priority=0):
    """
    Our main results...
    """

    # these are the 'good' settings now...
    HOST = '' # ML
    for seed in [1]:
        for env in ATARI_5:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority,
                'seed': seed,
            }
            # reference run to see how well we do without tuning...
            add_job(
                "DNA_FINAL",
                run_name=f"game={env} dna ({seed})",
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )
            add_job(
                "DNA_FINAL",
                run_name=f"game={env} ppo ({seed})",
                policy_epochs=4,
                value_epochs=0,
                distil_epochs=0,
                architecture='single',
                network='nature',
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
    PATH = f"DNA_EPOCHS (beta={1.0})"
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
            for policy_epochs in [1, 2, 3, 4]: # please add 3,4
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
            for distil_epochs in [0, 1, 2, 3, 4]: # please add 3, 4
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

def setup(priority_modifier=0):
    #dna_hps(0)     # done!
    #dna_noise(50)  # done!
    dna_beta(0)  # done!
    dna_epochs(-200) # ...
    dna_gae(0) # ...
    # dna_final(200)

    # bonus
    # dna_aux(0) # reward prediction as aux task...

