from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import TVF_reference_args, DNA_reference_args

ROLLOUT_SIZE = 128*128
ATARI_3_VAL = ['Breakout', 'WizardOfWor', 'YarsRevenge']
# ATARI_3_VAL = ['berzerk', 'boxing', 'zaxxon']
# ATARI_5_VAL = ['Bowling', 'Qbert', 'Berzerk', 'Boxing', 'Zaxxon']

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
    'use_tvf': False,

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
        process_up_to=16,
        envs=['Breakout', 'WizardOfWor', 'YarsRevenge'],
        hook=fixup_params,
        priority=priority,
    )




def setup(priority_modifier=0):
    dna_hps(priority_modifier)
