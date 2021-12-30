from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import PPO_reference_args, DNA_reference_args, TVF_reference_args

ROLLOUT_SIZE = 128*128
ATARI_5 = ['Centipede', 'CrazyClimber', 'Krull', 'SpaceInvaders', 'Zaxxon']  # Atari5

ERP1_args = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'architecture': 'dual',
    'export_video': False,
    'epochs': 50,
    'use_compression': "auto",
    'warmup_period': 1000,
    'disable_ev': False,
    'seed': 0,
    'mutex_key': "DEVICE",

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
    'distil_mini_batch_size': 512,
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
    'distil_batch_size': 1*ROLLOUT_SIZE,
    'distil_beta': 1.0,
    'distil_lr': 2.5e-4,
    'replay_mode': "uniform",
    'replay_hashing': True,
    'dna_dual_constraint': 0,

    # horizon
    'gamma': 0.99997,
    'tvf_gamma': 0.99997,
    'tvf_max_horizon': 30000,

    # other
    'observation_normalization': True, # pong (and others maybe) do not work without this, so jsut default it to on..

    'hostname': '',
}

# with 16x replay (needs compression though... so a little slower)
ERP16_args = ERP1_args.copy()
ERP16_args.update({
    'distil_period': 16,
    'replay_size':   16*ROLLOUT_SIZE,
    'distil_batch_size': 16*ROLLOUT_SIZE,
})


def replay_shadow(priority=0, hostname=''):
    """
    Run lots of replay buffers at the same time :)
    """
    # This is just to see how bad the duplication is if we train for a while...
    for env in ["Pong", "Breakout", "Alien", "CrazyClimber"][:2]: # just two for the moment (due to memory...)
        add_job(
            "RP_Shadow",
            env_name=env,
            run_name=f"{env}",
            epochs=50,
            default_params=ERP1_args,
            debug_replay_shadow_buffers=True,
            priority=priority,
            hostname=hostname,
        )


def replay_hashing(priority=0, hostname=''):
    """
    See if hashing increase the diversity of the replay.
    """
    # my guess is pong will not be very diverse, and hashing will help.
    for env in ["Pong", "Breakout", "Alien", "CrazyClimber"]:
        for mode in ["uniform", "sequential"]:
            add_job(
                "RP_Hashing",
                env_name=env,
                run_name=f"{env} (16{mode[0]})",
                replay_hashing=mode == "uniform",
                replay_mode=mode,
                epochs=10,
                default_params=ERP16_args,
                priority=priority,
                hostname=hostname,
            )
            add_job(
                "RP_Hashing",
                env_name=env,
                run_name=f"{env} (1{mode[0]})",
                replay_hashing=mode == "uniform",
                replay_mode=mode,
                epochs=10,
                default_params=ERP1_args,
                priority=priority,
                hostname=hostname,
            )

def reference_runs():
    """
    Reference runs for PPO, DNA, and TVF
    """
    for env in ['Breakout', 'CrazyClimber', 'SpaceInvaders']:
        add_job(
            "RP_Reference",
            env_name=env,
            run_name=f"{env} PPO",
            default_params=PPO_reference_args,
            priority=10,
            epochs=10, # enough for now...
            hostname='ML',
        )
        add_job(
            "RP_Reference",
            env_name=env,
            run_name=f"{env} DNA",
            default_params=DNA_reference_args,
            priority=10,
            epochs=10,  # enough for now...
            hostname='ML',
        )
        add_job(
            "RP_Reference",
            env_name=env,
            run_name=f"{env} TVF",
            default_params=TVF_reference_args,
            priority=10,
            epochs=10,  # enough for now...
            hostname='ML',
        )


def initial_random_search(priority=0):

    # would have been nice to do...
    # include gae 0.9, 0.95, 0.97
    # include sqrt...

    search_params = {
        # ppo params
        'distil_lr':         Categorical(1e-4, 2.5e-4, 5.0e-4),
        'distil_lr_anneal':  Categorical(True, False),
        'distil_epochs':     Categorical(1, 2, 3),
        'distil_period':     Categorical(1, 2, 4, 8, 16),
        'replay_size':       Categorical(*[x * ROLLOUT_SIZE for x in [0, 1, 2, 4, 8, 16]]),
        'distil_batch_size': Categorical(*[round(x * ROLLOUT_SIZE) for x in [0.25, 0.5, 1, 2, 4, 8]]),
        'dna_dual_constraint': Categorical(0, 0, 0, 0, 0.1, 0.3, 1.0, 3.0),
        'replay_mode':       Categorical("overwrite", "sequential", "uniform"), # req replay
        'replay_mixing':     Categorical(True, False), # req replay
    }

    main_params = ERP1_args.copy()
    # 10 is enough for pong, but I want to check if we eventually learn. (also verify loading and saving works)
    # 20 also lets me know if the algorithm can get to a 21 score or not (DC can sometimes cause issues).
    main_params["epochs"] = 10
    main_params["tvf_exp_gamma"] = 1.5 # just to make things a bit faster.
    main_params["hostname"] = ''
    del main_params["replay_hashing"] # first search did not support hashing

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

        # 2. make sure we don't process the distillation too much
        def get_total_compute():
            batch_size = params["distil_batch_size"]
            return batch_size * params["distil_epochs"] / params["distil_period"] / ROLLOUT_SIZE

        while get_total_compute() > 4:
            if params["distil_epochs"] > 1:
                params["distil_epochs"] -= 1
            else:
                params["distil_batch_size"] //= 2

        # 3. add a description
        code = str(params["distil_epochs"]) + str(params["distil_period"]) + str(params["replay_size"]//ROLLOUT_SIZE)
        if params["replay_size"] > 0:
            code += str(params["replay_mode"][0])
        params["description"] = code + " params:" + str(params)


    random_search(
        "RP_SEARCH",
        main_params,
        search_params,
        count=64,
        envs=['Breakout', 'CrazyClimber', 'SpaceInvaders'],
        hook=fixup_params,
        priority=priority,
    )


def setup(priority_modifier=0):
    # Initial experiments to make sure code it working, and find reasonable range for the hyperparameters.
    initial_random_search(priority=priority_modifier-5)
    replay_hashing(priority=priority_modifier+5, hostname="ML")
    reference_runs()
    replay_shadow(priority=priority_modifier+10, hostname="ML")
