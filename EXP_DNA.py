from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import TVF_reference_args, DNA_reference_args, ATARI_57
import numpy as np

"""
Todo:
1. Remove old settings
2. Upgrade all experiments to new code
3. Sequential job system?
4. Run main experiment (1 seed) make sure results match
5. Run 5M of all other experiments, make sure settings match
6. Confirm noise experiment

"""

ROLLOUT_SIZE = 128*128
ATARI_3_VAL = ['Assault', 'MsPacman', 'YarsRevenge']
ATARI_5 = ['BattleZone', 'DoubleDunk', 'NameThisGame', 'Phoenix', 'Qbert']

def add_run(experiment: str, run_name: str, default_args, env_args, seeds:int, subset:list, priority=0, **kwargs):

    args = STANDARD_ARGS.copy()
    args.update(default_args)
    args.update(env_args)

    for seed in range(1, seeds+1):
        for env in subset:
            add_job(
                experiment,
                run_name=f"game={env} {run_name} ({seed})",
                env_name=env,
                seed=seed,
                priority=priority - ((seed - 1) * 50),
                default_params=args,
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
STANDARD_ARGS = {
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
PPO_ORIG_ARGS = STANDARD_ARGS.copy()
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

DNA_TUNED_ARGS = STANDARD_ARGS.copy()
DNA_TUNED_ARGS.update({
    'gae_lambda': 0.8,
    'td_lambda': 0.95,
    'policy_epochs': 2,
    'value_epochs': 1,
    'distil_epochs': 2,
})

PPO_TUNED_ARGS = STANDARD_ARGS.copy()
PPO_TUNED_ARGS.update({
    'gae_lambda': 0.95,
    'td_lambda': 0.95,
    'policy_epochs': 1,
    'value_epochs': 0,
    'distil_epochs': 0,
    'architecture': 'single',
    'policy_network': 'nature_fat',
})

PPG_ARGS = STANDARD_ARGS.copy()
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

# old settings
# ---------------------------------------------------------------------------------------


# used in the PPO Paper
PPG_HARD_ARGS = STANDARD_ARGS.copy()
PPG_HARD_ARGS.update({
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
    'upload_batch': False,      # :(
})


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
PPG_HARD_ARGS = DNA_HARD_ARGS_HPS.copy()
PPG_HARD_ARGS.update({
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
    'upload_batch': False,      # :(

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


def dna_lambda(priority:int=0):
    """
    Demonstrate that GAE for advantages and for return values can be different.
    """
    for seed in [1, 2, 3, 4, 5]:
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': 'desktop',
                'priority': priority - (seed - 1) * 55,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': seed != 3, # need to know this on one seed at least
                'distil_beta': 1.0,
                'abs_mode': 'shadow' if seed == 4 else 'off',
            }
            for gae_lambda in [0.6, 0.8, 0.9, 0.95, 0.975]:
                for td_lambda in [0.8, 0.95]:
                    add_job(
                        "DNA_LAMBDA",
                        run_name=f"game={env} td_lambda={td_lambda} gae_lambda={gae_lambda} ({seed})",
                        gae_lambda=gae_lambda,
                        td_lambda=td_lambda,
                        policy_epochs=2,
                        value_epochs=1,
                        distil_epochs=2,
                        **args,
                        default_params=DNA_HARD_ARGS_HPS,
                    )
            # extra results added to check how td_lambda relates to noise.
            if seed == 4:
                for a in [0.95]:
                    for b in [0.6, 0.8, 0.9, 0.95, 0.975, 0.9875, 0.99375, 1.0]:
                        add_job(
                            "DNA_LAMBDA",
                            run_name=f"game={env} td_lambda={a} gae_lambda={b} ({seed})",
                            td_lambda=a,
                            gae_lambda=b,
                            policy_epochs=2,
                            value_epochs=1,
                            distil_epochs=2,
                            **args,
                            default_params=DNA_HARD_ARGS_HPS,
                        )
                        add_job(
                            "DNA_LAMBDA",
                            run_name=f"game={env} td_lambda={b} gae_lambda={a} ({seed})",
                            td_lambda=b,
                            gae_lambda=a,
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

def dna_A57(priority=0):

    for path, env_args in zip(["A57_HARD", "A57_EASY"], [HARD_MODE_ARGS, EASY_MODE_ARGS]):

        COMMON_ARGS = {
            'experiment': path,
            'seeds': 1,
            'subset': ATARI_57,
            'priority': 500 if path == "A57_HARD" else priority,
            'hostname': "" if path == "A57_HARD" else '',
        }

        add_run(
            run_name="dna_tuned",
            default_args=DNA_TUNED_ARGS,
            env_args=env_args,
            **COMMON_ARGS
        )

        add_run(
            run_name="ppo_tuned_fat",
            default_args=PPO_TUNED_ARGS,
            env_args=env_args,
            **COMMON_ARGS
        )

def dna_a57_old(priority=0):
    """
    Fill in the remaining 57 games for 1 seed.
    """

    for seed in [1]:
        for env in ATARI_57:
            args = {
                'env_name': env,
                'hostname': '',
                'priority': priority - (seed - 1) * 55,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': False,
                'distil_beta': 1.0,
            }
            add_job(
                "DNA_FINAL",
                run_name=f"game={env} dna_tuned ({seed})",
                gae_lambda=0.8,
                td_lambda=0.95,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

            args = {
                'env_name': env,
                'hostname': '',
                'priority': priority - (seed - 1) * 55,
                'seed': seed,
                'use_compression': False,
                'upload_batch': True,  # much faster
                'disable_ev': False,
                'distil_beta': 1.0,
                'terminal_on_loss_of_life': True,
                'reward_clipping': "off",
                'full_action_space': False,
                'repeat_action_probability': 0.0,
            }

            add_job(
                "DNA_FINAL_EASY",
                run_name=f"game={env} dna_tuned ({seed})",
                gae_lambda=0.8,
                td_lambda=0.95,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

def ppg_final(priority=0):
    """
    Our main results...
    (including ablations)
    """

    HOST = '' # ML

    for seed in [1, 2, 3]:
        for env in ATARI_5:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority - (seed - 1) * 55,
                'seed': seed,
                'epochs': 50,
                'anneal_target_epoch': 50,
            }

            add_job(
                "DNA_FINAL",
                run_name=f"game={env} ppg ({seed})",
                **args,
                default_params=PPG_HARD_ARGS,
            )


def dna_final_easy(priority=0):
    """
    Our main results...
    """

    HOST = '' # ML

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

                'terminal_on_loss_of_life': True,
                'reward_clipping': 1, # probably better to not use this... not an env setting...
                'full_action_space': False,
                'repeat_action_probability': 0.0,
            }

            # easy mode run...
            add_job(
                "DNA_FINAL_EASY",
                run_name=f"game={env} dna_tuned ({seed})",  # dna
                gae_lambda=0.8,
                td_lambda=0.95,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )



def dna_final(priority=0):
    """
    Our main results...
    """

    add_run(
        experiment="DNA_FINAL_EASY",
        run_name="dna_tuned",
        default_args=DNA_TUNED_ARGS,
        env_args=EASY_MODE_ARGS,
        priority=priority,
        seeds=3,
        subset=ATARI_5,
    )

    add_run(
        experiment="DNA_FINAL_EASY",
        run_name="ppo_tuned_fat",
        default_args=PPO_TUNED_ARGS,
        env_args=EASY_MODE_ARGS,
        priority=priority,
        seeds=3,
        subset=ATARI_5,
    )

    add_run(
        experiment="DNA_FINAL_EASY",
        run_name="ppo_orig",
        default_args=PPO_ORIG_ARGS,
        env_args=EASY_MODE_ARGS,
        priority=priority,
        seeds=3,
        subset=ATARI_5,
    )

    return


    HOST = '' # ML

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

                'terminal_on_loss_of_life': True,
                'reward_clipping': 1, # probably better to not use this... not an env setting...
                'full_action_space': False,
                'repeat_action_probability': 0.0,
            }

            # easy mode run...
            add_job(
                "DNA_FINAL_EASY",
                run_name=f"game={env} dna_tuned ({seed})",  # dna
                gae_lambda=0.8,
                td_lambda=0.95,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

            # add_job(
            #     "DNA_EASY",
            #     run_name=f"game={env} ppo_tuned_fat ({seed})",
            #     gae_lambda=0.95,
            #     td_lambda=0.95,
            #     policy_epochs=1,
            #     value_epochs=0,
            #     distil_epochs=0,
            #     architecture='single',
            #     network='nature_fat',
            #     **args,
            #     default_params=DNA_HARD_ARGS_HPS,
            # )
            # add_job(
            #     "DNA_EASY",
            #     run_name=f"game={env} ppo_orig ({seed})",
            #     **args,
            #     default_params=PPO_ORIG_ARGS,
            # )


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
                'reward_clipping': "off", # probably better to not use this... not an env setting...
                'full_action_space': False,
                'repeat_action_probability': 0.0,
            }

            # easy mode run...
            add_job(
                "DNA_FINAL_EASY",
                run_name=f"game={env} dna_tuned ({seed})",  # dna
                gae_lambda=0.8,
                td_lambda=0.95,
                policy_epochs=2,
                value_epochs=1,
                distil_epochs=2,
                **args,
                default_params=DNA_HARD_ARGS_HPS,
            )

            # add_job(
            #     "DNA_EASY",
            #     run_name=f"game={env} ppo_tuned_fat ({seed})",
            #     gae_lambda=0.95,
            #     td_lambda=0.95,
            #     policy_epochs=1,
            #     value_epochs=0,
            #     distil_epochs=0,
            #     architecture='single',
            #     network='nature_fat',
            #     **args,
            #     default_params=DNA_HARD_ARGS_HPS,
            # )
            # add_job(
            #     "DNA_EASY",
            #     run_name=f"game={env} ppo_orig ({seed})",
            #     **args,
            #     default_params=PPO_ORIG_ARGS,
            # )

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

            if seed == 1:
                # move to ablations...
                args['priority'] = +100
                add_job(
                    "DNA_FINAL",
                    run_name=f"game={env} ppo_orig ({seed})",
                    **args,
                    default_params=PPO_ORIG_ARGS,
                )

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
            }
            # ablations...
            # add_job(
            #     "DNA_ABLATION",
            #     run_name=f"game={env} ppo_orig ({seed})",
            #     **args,
            #     default_params=PPO_ORIG_ARGS,
            # )


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

def dna_tuning(priority=0):

    COMMON_ARGS = {
        'experiment': "DNA_TUNING",
        'seeds': 3,
        'subset': ATARI_3_VAL,
        'priority': priority,
        'hostname': "",
    }

    for epochs in [1, 2, 3, 4]:
        add_run(
            run_name=f"epochs=2{epochs}2",
            default_args=STANDARD_ARGS,
            env_args=HARD_MODE_ARGS,
            policy_epochs=2,
            value_epochs=epochs,
            distil_epochs=2,
            **COMMON_ARGS
        )


def dna_tuning_old(priority=0):

    # these value epochs where done first
    HOST = ''
    PATH = "DNA_TUNING"

    for seed in [1, 2, 3]: # three seeds is needed for an accurate results (but just do one for now...)
        for env in ATARI_3_VAL:
            args = {
                'env_name': env,
                'hostname': HOST,
                'priority': priority - (seed-1) * 10,
                'seed': seed,
                'disable_ev': seed == 3, # only ev on seed 3...
            }
            policy_epochs = 2
            distil_epochs = 2
            for value_epochs in [1, 2, 3, 4]:
                if value_epochs == 0:
                    if seed != 1:
                        continue
                add_job(
                    PATH,
                    run_name=f"game={env} epochs={policy_epochs}{value_epochs}{distil_epochs} ({seed})",
                    value_epochs=value_epochs,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )

    # one value epoch was, indeed, best...

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


def dna_distil(priority=0):
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
                    "DNA_DISTIL",
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

def ppo_tuning(priority:int = 0):

    # tuning for ppo
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
                    "PPO_TUNING",
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
                    "PPO_TUNING",
                    run_name=f"game={env} epochs={policy_epochs} lambda={gae_lambda} ({seed})",
                    policy_epochs=policy_epochs,
                    gae_lambda=gae_lambda,
                    td_lambda=gae_lambda,
                    **args,
                    default_params=DNA_HARD_ARGS_HPS,
                )




def ablations(priority:int = 0):

    COMMON_ARGS = {
        'experiment': "DNA_FINAL",
        'seeds': 3,
        'subset': ATARI_5,
        'priority': priority,
        'hostname': "",
        'env_args': HARD_MODE_ARGS,
    }

    # no seeds for ablations, just want to get an idea for what matters.

    # ---------------------------
    # DNA ablations

    add_run(
        run_name="dna_tuned_pmbs_512",
        default_args=DNA_TUNED_ARGS,
        policy_mini_batch_size=512,
        **COMMON_ARGS
    )

    # interesting, but ran out of time...
    # add_run(
    #     run_name="dna_fat",
    #     default_args=DNA_TUNED_ARGS,
    #     policy_network="nature_fat",
    #     value_network="nature_fat",
    #     **COMMON_ARGS
    # )

    add_run(
        run_name="dna_no_distil",
        default_args=DNA_TUNED_ARGS,
        distil_epochs=0,
        **COMMON_ARGS
    )

    add_run(
        run_name="dna_fixed_lambda",
        default_args=DNA_TUNED_ARGS,
        gae_lambda=0.95,
        td_lambda=0.95,
        **COMMON_ARGS
    )

    add_run(
        run_name="ppo_basic",
        default_args=PPO_TUNED_ARGS,
        policy_epochs=2,
        policy_network='nature',
        gae_lambda=0.95,
        td_lambda=0.95,
        policy_mini_batch_size=512,
        **COMMON_ARGS
    )

    # ---------------------------
    # PPO ablations

    add_run(
        run_name="ppo_2_fat",
        default_args=PPO_TUNED_ARGS,
        policy_epochs=2,
        **COMMON_ARGS
    )

    add_run(
        run_name="ppo_nature",
        default_args=PPO_TUNED_ARGS,
        policy_network="nature",
        **COMMON_ARGS
    )

    add_run(
        run_name="ppo_lambda",
        default_args=PPO_TUNED_ARGS,
        gae_lambda=0.8,
        td_lambda=0.95,
        **COMMON_ARGS
    )

    add_run(
        run_name="ppo_orig",
        default_args=PPO_ORIG_ARGS,
        **COMMON_ARGS
    )

    # ---------------------------
    # PPG ablations

    add_run(
        run_name="ppg",
        default_args=PPG_HARD_ARGS,
        **COMMON_ARGS
    )

    add_run(
        run_name="ppg_tuned",
        default_args=PPG_HARD_ARGS,
        policy_epochs=2,
        value_epochs=1,
        aux_epochs=2,
        **COMMON_ARGS
    )



def verify(run_list):
    """
    Check two sets of experiments, check if they match exactly or not.
    Useful for confirming that the changes I made to config did not result in differences to the algorithm

    Process:
    - code up the verification code
    - run on one thing and see if we match
    - do a code clean up on reference code from dna_final
    - switch all experiments over to new config system
    - implement sequential run system?
    - run verification to 5m or something

    What this gives me
    - proof that the replication script runs
    - proof that the settings on the replication script are right
    - proof that results are not due to config errors.

    also...
    - verify that PPG is working properly
    - verify that experimental settings (as run) were correct for all main experiments (make a checklist for this)

    """
    pass


def setup(priority_modifier=0):


    # ---------------------------
    # atari-3 validation

    # dna_hps()

    # ppo_tuning()
        #dna_tuning()
    #
    # dna_noise()
    # dna_distil()
        #dna_lambda()

    # ---------------------------
    # atari-5 test set

    dna_final(50)
    # ppg_final(0)

    # ablations(600)

    # ---------------------------
    # atari-57
    dna_A57(-500)

    pass


