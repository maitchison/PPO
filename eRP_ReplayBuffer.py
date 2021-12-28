from runner_tools import WORKERS, MYSTIC_FIVE, add_job, random_search, Categorical

# changes to search
# include distil_mini_batch_size
# wider DC / balanced no DC?
# simple only (to get it working...)

ROLLOUT_SIZE = 128*128
ATARI_5 = ['Centipede', 'CrazyClimber', 'Krull', 'SpaceInvaders', 'Zaxxon']  # Atari5

RP_args = {
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
    'distil_resampling': False,
    'distil_batch_size': ROLLOUT_SIZE//2,
    'distil_beta': 1.0,
    'distil_lr': 2.5e-4,
    'replay_mode': "uniform",
    'replay_mixing': False,
    'dna_dual_constraint': 0.3,

    # horizon
    'gamma': 0.99997,
    'tvf_gamma': 0.99997,
    'tvf_max_horizon': 30000,

    'hostname': '',
}

# stub
E2_args = RP_args


def initial_random_search(priority=0):

    # would have been nice to do...
    # include gae 0.9, 0.95, 0.97
    # include sqrt...

    search_params = {
        # ppo params
        'distil_lr':         Categorical(1e-4, 2.5e-4, 5.0e-4),
        'distil_lr_anneal':  Categorical(True, False),
        'distil_epochs':     Categorical(1, 2, 4, 8),
        'distil_period':     Categorical(1, 2, 4, 8),
        'replay_size':       Categorical(*[x * ROLLOUT_SIZE for x in [0, 1, 2, 4, 8, 16]]),
        'distil_batch_size': Categorical(*[round(x * ROLLOUT_SIZE) for x in [0.25, 0.5, 1, 2, 4]]),
        'dna_dual_constraint': Categorical(0, 0, 0, 0, 0.1, 0.3, 1.0, 3.0),
        'replay_mode':       Categorical("overwrite", "sequential", "uniform"), # req replay
        'replay_mixing':     Categorical(True, False), # req replay
        'distil_resampling': Categorical(True, False),
        # other stuff
        'observation_normalization': Categorical(True, False),
        'observation_scaling': Categorical("centered", "unit"),
        'layer_norm': Categorical(True, False),
        # tvf
        'use_tvf': Categorical(True, False),
    }

    main_params = RP_args.copy()
    # 10 is enough for pong, but I want to check if we eventually learn. (also verify loading and saving works)
    # 20 also lets me know if the algorithm can get to a 21 score or not (DC can sometimes cause issues).
    main_params["epochs"] = 10
    main_params["hostname"] = 'ML'

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

def regression_tests():
    # for a while pong was terriable, need to go back and see what happened.
    # big question is, is it a code problem, or a settings problem?

    for seed in range(3):
        for env in ["Pong", "Breakout"]:
            for norming in [True, False]:
                for scaling in [True, False]:
                    add_job(
                        "RP_Seed_features",
                        env_name=env,
                        run_name=f"game={env} norm={norming} scale={scaling} seed={seed}",
                        replay_size=0,
                        dna_dual_constraint=0.0,

                        observation_normalization=norming,
                        observation_scaling="centered" if scaling else "unit",

                        distil_batch_size=ROLLOUT_SIZE,
                        tvf_force_ext_value_distil=True,
                        default_params=E2_args,
                        epochs=10,
                        seed=seed,
                        priority=25 if seed == 0 else seed,
                    )
                    add_job(
                        "RP_Seed_full",
                        env_name=env,
                        run_name=f"game={env} norm={norming} scale={scaling} seed={seed}",
                        replay_size=0,
                        dna_dual_constraint=0.0,

                        observation_normalization=norming,
                        observation_scaling="centered" if scaling else "unit",

                        distil_batch_size=ROLLOUT_SIZE,
                        tvf_force_ext_value_distil=False,
                        default_params=E2_args,
                        epochs=10,
                        seed=seed,
                        priority=25 if seed == 0 else seed,
                    )

    for seed in range(3):
        for env in ["Pong", "Breakout"]:
            for norming in [True, False]:
                for scaling in [True, False]:
                    add_job(
                        "RP_Seed_layernorm",
                        env_name=env,
                        run_name=f"game={env} norm={norming} scale={scaling} [layer_norm] seed={seed}",
                        replay_size=0,
                        dna_dual_constraint=0.0,

                        observation_normalization=norming,
                        observation_scaling="centered" if scaling else "unit",
                        layer_norm=True,

                        distil_batch_size=ROLLOUT_SIZE,
                        tvf_force_ext_value_distil=True,
                        default_params=E2_args,
                        epochs=10,
                        seed=seed,
                        priority=25 if seed == 0 else 0,
                    )

    for seed in range(3):

        env = "Pong"
        add_job(
            "RP_Seed_veryold",
            env_name=env,
            run_name=f"game={env} reference seed={seed}",
            replay_size=0,
            dna_dual_constraint=0.0,
            observation_normalization=False,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            distil_batch_size=None,
            mutex_key=None,
            distil_mini_batch_size=None,
            distil_resampling=None,
            epochs=10,
            priority=20
        )

        for env in ["Pong", "Breakout"]:
            add_job(
                "RP_Seed_k",
                env_name=env,
                run_name=f"game={env} on seed={seed}",
                replay_size=0,
                dna_dual_constraint=0.0,
                observation_normalization=True,
                distil_batch_size=ROLLOUT_SIZE,
                tvf_force_ext_value_distil=True,
                default_params=E2_args,
                epochs=10,
                priority=10
            )

            # add_job(
            #     "RP_Seed_on",
            #     env_name=env,
            #     run_name=f"game={env} on seed={seed}",
            #     replay_size=0,
            #     dna_dual_constraint=0.0,
            #     observation_normalization=True,
            #     distil_batch_size=ROLLOUT_SIZE,
            #     tvf_force_ext_value_distil=True,
            #     default_params=E2_args,
            #     epochs=10,
            #     priority=10
            # )


            add_job(
                "RP_Seed_on",
                env_name=env,
                run_name=f"game={env} on seed={seed}",
                replay_size=0,
                dna_dual_constraint=0.0,
                observation_normalization=True,
                distil_batch_size=ROLLOUT_SIZE,
                tvf_force_ext_value_distil=True,
                default_params=E2_args,
                epochs=10,
                priority=10
            )

            add_job(
                "RP_Seed_k",
                env_name=env,
                run_name=f"game={env} no_time seed={seed}",
                replay_size=0,
                dna_dual_constraint=0.0,
                tvf_time_scale="zero",
                distil_batch_size=ROLLOUT_SIZE,
                tvf_force_ext_value_distil=True,
                default_params=E2_args,
                epochs=10,
                priority=10
            )

            add_job(
                "RP_Seed_k",
                env_name=env,
                run_name=f"game={env} no_time simple_head seed={seed}",
                replay_size=0,
                dna_dual_constraint=0.0,
                tvf_time_scale="zero",
                tvf_hidden_units=1,
                distil_batch_size=ROLLOUT_SIZE,
                tvf_force_ext_value_distil=True,
                default_params=E2_args,
                epochs=10,
                priority=15
            )




            add_job(
                "RP_Seed_k",
                env_name=env,
                run_name=f"game={env} de=2 seed={seed}",
                replay_size=0,
                dna_dual_constraint=0.0,
                distil_epochs=2,
                distil_batch_size=ROLLOUT_SIZE,
                tvf_force_ext_value_distil=True,
                default_params=E2_args,
                epochs=10,
                priority=10
            )
            add_job(
                "RP_Seed_k",
                env_name=env,
                run_name=f"game={env} de=3 seed={seed}",
                replay_size=0,
                dna_dual_constraint=0.0,
                distil_epochs=3,
                distil_batch_size=ROLLOUT_SIZE,
                tvf_force_ext_value_distil=True,
                default_params=E2_args,
                epochs=10,
                priority=10
            )
            add_job(
                "RP_Seed_k",
                env_name=env,
                run_name=f"game={env} reference seed={seed}",
                replay_size=0,
                dna_dual_constraint=0.0,
                observation_normalization=False,
                distil_batch_size=ROLLOUT_SIZE,
                tvf_force_ext_value_distil=True,
                default_params=E2_args,
                epochs=10,
                priority=10
            )


        add_job(
            "RP_Seed",
            env_name='Pong',
            run_name=f"game=Pong ref seed={seed}",
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=10,
            seed=seed,
        )

    for seed in range(4):
        add_job(
            "RP_Seed_var",
            env_name='Pong',
            run_name=f"game=Pong ref seed={seed}",
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=10,
            seed=seed,
        )



        for seed in range(4):
            add_job(
                "RP_Seed_var_boost",
                env_name='Pong',
                run_name=f"game=Pong ref seed={seed}",
                replay_size=0,
                distil_var_boost=0.5,
                dna_dual_constraint=0.0,
                distil_batch_size=ROLLOUT_SIZE,
                tvf_force_ext_value_distil=True,
                default_params=E2_args,
                epochs=10,
                priority=10,
                seed=seed,
            )

    for seed in range(4):
        add_job(
            "RP_Seed_head_boost_x10",
            env_name='Pong',
            run_name=f"game=Pong ref seed={seed}",
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=10,
            seed=seed,
        )

    for seed in range(4):
        add_job(
            "RP_Seed_head_boost_x100_min",
            env_name='Pong',
            run_name=f"game=Pong ref seed={seed}",
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_min_var=0.1,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=10,
            seed=seed,
        )

    for seed in range(4):
        add_job(
            "RP_Seed_head_boost_x100",
            env_name='Pong',
            run_name=f"game=Pong ref seed={seed}",
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=10,
            seed=seed,
        )

    for seed in range(4):
        add_job(
            "RP_Seed_DC",
            env_name='Pong',
            run_name=f"game=Pong dc seed={seed}",
            replay_size=0,
            dna_dual_constraint=1.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=10,
            seed=seed,
        )


    for experiment_name in ["RP_Test_good", "RP_test_old", "RP_delay"]:
        add_job(
            experiment_name,
            env_name='Pong',
            run_name=f"game=Pong reference",
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=10,
        )
        add_job(
            experiment_name,
            env_name='Pong',
            run_name=f"game=Pong dc",
            replay_size=0,
            dna_dual_constraint=0.3,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=10,
        )
        add_job(
            experiment_name,
            env_name='Pong',
            run_name=f"game=Pong no_policy",
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            policy_epochs=0,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=10,
        )
        add_job(
            experiment_name,
            env_name='Pong',
            run_name=f"game=Pong low_mbs",
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            distil_mini_batch_size=128, # add some stocasticity, see if that helps...
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=10,
        )
        add_job(
            experiment_name,
            env_name='Pong',
            run_name=f"game=Pong 210",
            distil_epochs=2,
            distil_period=1,
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=0,
        )
        add_job(
            experiment_name,
            env_name='Pong',
            run_name=f"game=Pong 120",
            distil_epochs=1,
            distil_period=2,
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=0,
        )
        add_job(
            experiment_name,
            env_name='Pong',
            run_name=f"game=Pong 220",
            distil_epochs=2,
            distil_period=2,
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=0,
        )
        add_job(
            experiment_name,
            env_name='Pong',
            run_name=f"game=Pong 440",
            distil_epochs=4,
            distil_period=4,
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=0,
        )
        add_job(
            experiment_name,
            env_name='Pong',
            run_name=f"game=Pong 880",
            distil_epochs=8,
            distil_period=8,
            replay_size=0,
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=0,
        )
        add_job(
            experiment_name,
            env_name='Pong',
            run_name=f"game=Pong 121u",
            distil_epochs=1,
            distil_period=2,
            replay_size=ROLLOUT_SIZE,
            replay_mode="uniform",
            dna_dual_constraint=0.0,
            distil_batch_size=ROLLOUT_SIZE,
            tvf_force_ext_value_distil=True,
            default_params=E2_args,
            epochs=10,
            priority=0,
        )


def setup(priority_modifier=0):

    # Initial experiments to make sure code it working, and find reasonable range for the hyperparameters.
    initial_random_search(priority=priority_modifier-5)

    regression_tests()

    # This is the reference run, I know this works...
    add_job(
        f"RP_Test",
        env_name='Pong',
        run_name=f"game=Pong reference",
        replay_size=0,
        dna_dual_constraint=0.0,
        distil_batch_size=ROLLOUT_SIZE,
        tvf_force_ext_value_distil=True,
        default_params=E2_args,
        epochs=10,
        priority=0,
    )
    add_job(
        f"RP_Test",
        env_name='Pong',
        run_name=f"game=Pong complex no_replay",
        replay_size=0,
        dna_dual_constraint=0.0,
        distil_batch_size=ROLLOUT_SIZE,
        tvf_force_ext_value_distil=False,
        default_params=E2_args,
        epochs=10,
        priority=0,
    )
    add_job(
        f"RP_Test",
        env_name='Pong',
        run_name=f"game=Pong simple 111u",
        distil_epochs=1,
        distil_period=1,
        replay_size=ROLLOUT_SIZE,
        dna_dual_constraint=0.0,
        distil_batch_size=ROLLOUT_SIZE,
        tvf_force_ext_value_distil=True,
        default_params=E2_args,
        epochs=10,
        priority=0,
    )
    add_job(
        f"RP_Test",
        env_name='Pong',
        run_name=f"game=Pong simple 121u",
        distil_epochs=1,
        distil_period=2,
        replay_size=ROLLOUT_SIZE,
        dna_dual_constraint=0.0,
        distil_batch_size=ROLLOUT_SIZE,
        tvf_force_ext_value_distil=True,
        default_params=E2_args,
        hostname="",
        epochs=10,
        priority=0,
    )
    add_job(
        f"RP_Test",
        env_name='Pong',
        run_name=f"game=Pong simple 121u dc=0.3",
        distil_epochs=1,
        distil_period=2,
        replay_size=ROLLOUT_SIZE,
        dna_dual_constraint=0.3,
        distil_batch_size=ROLLOUT_SIZE,
        tvf_force_ext_value_distil=True,
        default_params=E2_args,
        epochs=10,
        priority=0,
    )

    #for env in MYSTIC_FIVE:
    # for env in []:
    #
    #     # check out DC when full curve is on.
    #     # full curve might need a much stronger constraint?
    #     for dc in [0, 0.1, 0.3, 1.0, 3.0, 10.0]:
    #         # still not sure which one of these is best?
    #         add_job(
    #             f"E2_DistilBatchSize",
    #             env_name=env,
    #             run_name=f"game={env} dc={dc} bs={1}",
    #             replay_size=round(ROLLOUT_SIZE),
    #             default_params=E2_args,
    #             dna_dual_constraint=dc,
    #             epochs=10 if env == "Pong" else 0,
    #             priority=priority_modifier,
    #         )
    #
    #     # I should also take a look at the period soon...
    #     for dp in [1, 2, 4]:
    #         for de in [1, 2, 4]:
    #             for dbs in [1, 2, 4]:
    #                 add_job(
    #                     f"E2_ReplaySize",
    #                     env_name=env,
    #                     run_name=f"game={env} dp={dp} de={de} dbs={dbs} rs=16x",
    #                     distil_epochs=de,
    #                     distil_period=dp,
    #                     replay_size=16*ROLLOUT_SIZE,
    #                     distil_batch_size=round(dbs*ROLLOUT_SIZE),
    #                     default_params=E2_args,
    #                     epochs=10 if env == "Pong" else 0,
    #                     priority=priority_modifier,
    #                 )
    #
    #     for rs in [0, 1, 2, 4, 8, 16, 32]:
    #         add_job(
    #             f"E2_ReplaySize",
    #             env_name=env,
    #             run_name=f"game={env} rs={rs}",
    #             replay_size=rs*ROLLOUT_SIZE,
    #             default_params=E2_args,
    #             epochs=10 if env == "Pong" else 0,
    #             priority=priority_modifier,
    #         )
