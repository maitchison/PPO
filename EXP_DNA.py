from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import TVF_reference_args, DNA_reference_args

ROLLOUT_SIZE = 128*128
ATARI_3_VAL = ['NameThisGame', 'WizardOfWor', 'YarsRevenge']
ATARI_5 = ['Asterix', 'BattleZone', 'DoubleDunk', 'Phoenix', 'RiverRaid']

# updated args tuned for hard mode
DNA_HARD_ARGS = {
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
    'agents': 256,
    'n_steps': 64,
    'policy_mini_batch_size': 512,
    'value_mini_batch_size': 1024,
    'distil_mini_batch_size': 1024,
    'policy_epochs': 2,
    'value_epochs': 1,
    'distil_epochs': 1, # 4 passes total
    'ppo_epsilon': 0.1,
    'policy_lr': 2.5e-4,
    'value_lr': 2.5e-4,
    'distil_lr': 2.5e-4,
    'entropy_bonus': 1e-2, # increased entropy due to full action space
    'hidden_units': 256,
    'gae_lambda': 0.95,
    'td_lambda': 0.95,

    # tvf params
    'use_tvf': False,

    # distil / replay buffer (This would have been called h11 before
    'distil_period': 1,
    'replay_size': 32*1024,
    'distil_batch_size': 16*1024,
    'distil_beta': 1.0,

    'replay_mode': "uniform",

    'dna_dual_constraint': 0,

    # horizon
    'gamma': 0.997,

    # other
    'observation_normalization': True, # pong (and others maybe) do not work without this, so jsut default it to on..

    'hostname': '',
}

def dna_hps(priority: int = 0):

    search_params = {
        # ppo params
        'entropy_bonus':     Categorical(3e-2, 1e-2, 3e-3),
        'agents':            Categorical(64, 128, 256),
        'n_steps':           Categorical(64, 128, 256),
        'gamma':             Categorical(0.99, 0.997, 0.999, 0.9997),
        'gae_lambda':        Categorical(0.9, 0.95, 0.975, 0.9875),
        # dna params
        'policy_lr':         Categorical(1e-4, 2.5e-4, 5e-4),
        'distil_lr':         Categorical(1e-4, 2.5e-4, 5e-4),
        'value_lr':          Categorical(1e-4, 2.5e-4, 5e-4),
        'td_lambda':         Categorical(0.9, 0.95, 0.975, 0.9875),
        'policy_epochs':     Categorical(1, 2, 3),
        'value_epochs':      Categorical(1, 2, 3),
        'distil_epochs':     Categorical(1, 2, 3),
        'policy_mini_batch_size': Categorical(512, 1024, 2048),
        'value_mini_batch_size': Categorical(512, 1024, 2048),
        'distil_mini_batch_size': Categorical(512, 1024, 2048),
        'replay_size':       Categorical(*[x * ROLLOUT_SIZE for x in [1, 2, 4]]),
        'distil_batch_size': Categorical(*[round(x * ROLLOUT_SIZE) for x in [0.5, 1, 2]]),
        'repeated_action_penalty': Categorical(0, 0.25, 1.0),
        'entropy_scaling':   Categorical(True, False),

        # replay params
        'replay_mode':       Categorical("overwrite", "sequential", "uniform", "off"),
    }

    main_params = DNA_HARD_ARGS.copy()

    main_params["epochs"] = 50
    main_params["hostname"] = ''

    def fixup_params(params):

        # 1. make sure distil_batch_size does not exceed the replay buffer size.
        using_replay = params["replay_size"] > 0
        if using_replay:
            # cap batch size to replay size
            max_batch_size = params["replay_size"]
        else:
            # cap batch size to rollout size
            max_batch_size = ROLLOUT_SIZE
        params["distil_batch_size"] = min(params["distil_batch_size"], max_batch_size)

        params["use_compression"] = True


    random_search(
        "DNA_SEARCH",
        main_params,
        search_params,
        count=64,
        process_up_to=32,
        envs=['NameThisGame', 'WizardOfWor', 'YarsRevenge'],
        hook=fixup_params,
        priority=priority,
    )


def dna_replay(priority=0):
    HOST = ''
    ROLLOUT_SIZE = 16*1024

    for seed in [1]: # this will need 3 seeds (results are close)
        for env in ATARI_5:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority + (50 if seed == 1 else 0),
                'seed': seed,
                'use_compression': True,
            }
            add_job(
                "DNA_REPLAY",
                run_name=f"game={env} replay=2u ({seed})",
                replay_size=2*ROLLOUT_SIZE,
                replay_mode="uniform",
                **args,
                default_params=DNA_HARD_ARGS,
            )

            add_job(
                "DNA_REPLAY",
                run_name=f"game={env} replay=2u de=3 ({seed})",
                replay_size=2 * ROLLOUT_SIZE,
                replay_mode="uniform",
                distil_epochs=3,
                **args,
                default_params=DNA_HARD_ARGS,
            )
            add_job(
                "DNA_REPLAY",
                run_name=f"game={env} replay=2u de=3 mixing ({seed})",
                replay_size=2 * ROLLOUT_SIZE,
                replay_mode="uniform",
                distil_epochs=3,
                replay_mixing=True,
                **args,
                default_params=DNA_HARD_ARGS,
            )
            add_job(
                "DNA_REPLAY",
                run_name=f"game={env} replay=16u ({seed})",
                replay_size=16 * ROLLOUT_SIZE,
                replay_mode="uniform",
                **args,

                default_params=DNA_HARD_ARGS,
            )
            add_job(
                "DNA_REPLAY",
                run_name=f"game={env} replay=16x ({seed})",
                replay_size=16*ROLLOUT_SIZE,
                replay_mode="sequential",
                **args,
                default_params=DNA_HARD_ARGS,
            )
            add_job(
                "DNA_REPLAY",
                run_name=f"game={env} replay=2x ({seed})",
                replay_size=2 * ROLLOUT_SIZE,
                replay_mode="sequential",
                **args,
                default_params=DNA_HARD_ARGS,
            )
            add_job(
                "DNA_REPLAY",
                run_name=f"game={env} replay=off ({seed})",
                replay_size=0 * ROLLOUT_SIZE,
                replay_mode="sequential",

                **args,
                default_params=DNA_HARD_ARGS,
            )
            add_job(
                "DNA_REPLAY",
                run_name=f"game={env} replay=off de=3 ({seed})",
                replay_size=0 * ROLLOUT_SIZE,
                replay_mode="sequential",
                distil_epoch=3,
                **args,
                default_params=DNA_HARD_ARGS,
            )
            add_job(
                "DNA_REPLAY",
                run_name=f"game={env} replay=ppg ({seed})",
                replay_size=16 * ROLLOUT_SIZE,
                replay_mode="sequential",
                distil_period=16,
                distil_batch_size=16*16*1024, # huge batch... but only every 16 updates
                **args,
                default_params=DNA_HARD_ARGS,
            )

    # just want to see if increased distilation helps here?
    # for seed in [1]:
    #     for env in ATARI_3_VAL:
    #         args = {
    #             'env_name': env,
    #             'hostname': HOST,
    #             'priority': priority + (50 if seed == 1 else 0) - 100,
    #             'seed': seed,
    #             'distil_epochs': 3,
    #             'use_compression': True,
    #         }
    #         add_job(
    #             "DNA_REPLAY2",
    #             run_name=f"game={env} replay=2u ({seed})",
    #             replay_size=2*ROLLOUT_SIZE,
    #             replay_mode="uniform",
    #             **args,
    #             default_params=DNA_HARD_ARGS,
    #         )
    #         add_job(
    #             "DNA_REPLAY2",
    #             run_name=f"game={env} replay=16x ({seed})",
    #             replay_size=16*ROLLOUT_SIZE,
    #             replay_mode="sequential",
    #             **args,
    #             default_params=DNA_HARD_ARGS,
    #         )
    #         add_job(
    #             "DNA_REPLAY2",
    #             run_name=f"game={env} replay=2x ({seed})",
    #             replay_size=2 * ROLLOUT_SIZE,
    #             replay_mode="sequential",
    #             **args,
    #             default_params=DNA_HARD_ARGS,
    #         )
    #         add_job(
    #             "DNA_REPLAY2",
    #             run_name=f"game={env} replay=off ({seed})",
    #             replay_size=0 * ROLLOUT_SIZE,
    #             replay_mode="sequential",
    #             **args,
    #             default_params=DNA_HARD_ARGS,
    #         )


def dna_distil(priority=0):
    HOST = ''
    for seed in [1]:
        for env in ATARI_5:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority,
                'seed': seed,
            }
            add_job(
                "DNA_DISTIL",
                run_name=f"game={env} distil_on ({seed})",
                **args,
                default_params=DNA_HARD_ARGS,
            )
            add_job(
                "DNA_DISTIL",
                run_name=f"game={env} distil_off ({seed})",
                distil_epochs=0,
                policy_epochs=3,
                **args,
                default_params=DNA_HARD_ARGS,
            )
            add_job(
                "DNA_DISTIL",
                run_name=f"game={env} ppo ({seed})",
                policy_epochs=4,
                distil_epochs=0,
                architecture='single',
                network='nature_fat',
                **args,
                default_params=DNA_HARD_ARGS,
            )


def dna_axis(priority=0):

    # quick axis search

    HOST = ''

    for seed in [1]:
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority,
                'seed': seed,
            }
            for beta in [0.3, 1, 3.0]:
                add_job(
                    "DNA_AXIS",
                    run_name=f"game={env} distil_beta={beta} ({seed})",
                    distil_beta=beta,
                    **args,
                    default_params=DNA_HARD_ARGS,
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
            }
            add_job(
                "DNA_NOISE",
                run_name=f"game={env} ({seed})",
                abs_mode='shadow',
                **args,
                default_params=DNA_HARD_ARGS,
            )

def setup(priority_modifier=0):
    dna_hps(priority_modifier-50)
    dna_axis(priority_modifier + 50)
    dna_replay(priority_modifier + 50)
    dna_distil(priority_modifier + 0)
    dna_noise(priority_modifier + 200)