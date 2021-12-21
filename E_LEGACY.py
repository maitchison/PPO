"""
Legacy experiments, these should be deleted as they used the old code.
"""

from runner_tools import *

def setup_DNA_Atari57():

    default_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'architecture': 'dual',
        'export_video': True,
        'epochs': 50,  # really want to see where these go...
        'use_compression': 'auto',
        'warmup_period': 1000,  # helps on some games to make sure they are really out of sync at the beginning of training.
        'disable_ev': False,
        'seed': 0,

        # env parameters
        'time_aware': False,
        'terminal_on_loss_of_life': True,  # to match rainbow
        'reward_clipping': 1,  # to match rainbow

        # parameters found by hyperparameter search...
        'max_grad_norm': 5.0,
        'agents': 128,
        'n_steps': 256,
        'policy_mini_batch_size': 256,
        'value_mini_batch_size': 256,
        'policy_epochs': 4,
        'value_epochs': 2,
        'distill_epochs': 3,
        'distill_beta': 0.5,
        'target_kl': 0.03,
        'ppo_epsilon': 0.2,
        'policy_lr': 1e-4,
        'value_lr': 2.5e-4,
        'distill_lr': 1e-4,
        'entropy_bonus': 1e-3,
        'tvf_force_ext_value_distill': True,
        'hidden_units': 128,
        'value_transform': 'sqrt',
        'gae_lambda': 0.95,

        # tvf params
        'use_tvf': False,

        # horizon
        'gamma': 0.99997,
    }

    for env in canonical_57:
        if env in ["Surround"]:
            continue

        add_job(
            f"DNA_Atari57",
            env_name=env,
            run_name=f"{env}_best",
            default_params=default_args,
            priority=50,
            hostname='ML-Rig',
        )


def setup_value_scale():

    """
    Experiments to see if learning value * f(h) is better than learning value directly.
    """
    for env in ["Krull", "Breakout", "Seaquest", "CrazyClimber"]:
        for tvf_value_scale_fn in ["identity", "linear", "log", "sqrt"]:
            add_job(
                f"ValueScale_{env}",
                env_name=env,
                run_name=f"fn={tvf_value_scale_fn}",
                tvf_value_scale_fn=tvf_value_scale_fn,
                default_params=standard_args,
                epochs=50,
                priority=0,
                seed=2, # check if old krull result was a fluke... ?
                hostname='',
            )

def setup_gamma():
    """
    Experiments to see how stable algorithms are with undiscounted rewards.
    """

    # check rediscounting on a number of games
    for env in ["Krull", "Breakout", "Freeway", "Hero", "CrazyClimber"]:
        for gamma in [0.99, 1.0]:
            for tvf_gamma in [0.99, 1.0]:
                if gamma == 1.0 and tvf_gamma == 0.99:
                    # this combination doesn't work as we end up multiplying noise by 1e131
                    continue
                add_job(
                    f"Gamma_{env}",
                    env_name=env,
                    run_name=f"algo=TVF gamma={gamma} tvf_gamma={tvf_gamma}",
                    gamma=gamma,
                    tvf_gamma=tvf_gamma,
                    default_params=standard_args,
                    epochs=30,
                    priority=0,
                    hostname='',
                )
            # for reference
            add_job(
                f"Gamma_{env}",
                env_name=env,
                run_name=f"algo=DNA gamma={gamma}",
                use_tvf=False,
                gamma=gamma,
                default_params=standard_args,
                epochs=30,
                priority=50,
                hostname='',
            )
            # for reference
            add_job(
                f"Gamma_{env}",
                env_name=env,
                run_name=f"algo=PPO gamma={gamma}",
                use_tvf=False,
                gamma=gamma,
                architecture='single',
                default_params=standard_args,
                epochs=30,
                priority=0,
                hostname='',
            )


def E01():
    """
    E1.1: Show different games require different gamma (variance / bias tradeoff)
    Should take around 2-3 days to complete.
    """
    for gamma in [0.9, 0.99, 0.999, 0.9997, 0.9999, 0.99997, 1.0]:
        for env in DIVERSE_10:
            for run in [1]: # just one run for the moment...
                add_job(
                    f"E11_PerGameGamma",
                    env_name=env,
                    run_name=f"game={env} gamma={gamma} (seed={run})",
                    use_tvf=False,
                    gamma=gamma,
                    architecture='single',
                    default_params=standard_args,
                    epochs=50,
                    priority=0,
                    seed=run,
                    hostname='',
                )

    # special case with pong to see if we can get it to work with better curve quality
    add_job(
        f"E11_PerGameGamma",
        env_name="Pong",
        run_name=f"game=Pong tvf_adv (seed=1)",
        use_tvf=True,
        gamma=0.99997,
        tvf_gamma=0.99997,
        default_params=enhanced_args,
        epochs=20,
        priority=100,
        seed=1,
        hostname='',
    )

    # special case with pong to see if we can get it to work with better curve quality
    add_job(
        f"E11_PerGameGamma",
        env_name="Pong",
        run_name=f"game=Pong tvf_simple_dist (seed=1)",
        use_tvf=True,
        tvf_force_ext_value_distill=True, # important if value function curve very difficult to learn
        gamma=0.99997,
        tvf_gamma=0.99997,
        default_params=enhanced_args,
        epochs=20,
        priority=200,
        seed=1,
        hostname='',
    )

    # special case with pong to see if we can get it to work with better curve quality
    add_job(
        f"E11_PerGameGamma",
        env_name="Pong",
        run_name=f"game=Pong tvf_tight_dist (seed=1)",
        use_tvf=True,
        tvf_force_ext_value_distill=False,
        distill_beta=10.0,
        gamma=0.99997,
        tvf_gamma=0.99997,
        default_params=enhanced_args,
        epochs=20,
        priority=200,
        seed=1,
        hostname='',
    )

    # special case with pong to see if we can get it to work with better curve quality
    add_job(
        f"E11_PerGameGamma",
        env_name="Pong",
        run_name=f"game=Pong tvf_no_dist (seed=1)",
        use_tvf=True,
        distill_epochs=0,
        gamma=0.99997,
        tvf_gamma=0.99997,
        default_params=enhanced_args,
        epochs=20,
        priority=200,
        seed=1,
        hostname='',
    )

    # another special case with pong to see if we can get it to work with better curve quality
    for tvf_mode in ['nstep', 'adaptive']:
        add_job(
            f"E11_PerGameGamma",
            env_name="Pong",
            run_name=f"game=Pong tvf_{tvf_mode} (seed=1)",
            use_tvf=True,
            gamma=0.99997,
            tvf_gamma=0.99997,
            tvf_mode=tvf_mode,
            default_params=enhanced_args,
            epochs=20,
            priority=200,
            seed=1,
            hostname='',
        )
    add_job(
        f"E11_PerGameGamma",
        env_name="Pong",
        run_name=f"game=Pong tvf_masked (seed=1)",
        use_tvf=True,
        gamma=0.99997,
        tvf_gamma=0.99997,
        tvf_exp_mode="masked",
        default_params=enhanced_args,
        epochs=20,
        priority=200,
        seed=1,
        hostname='',
    )

    # may as well see how TVF goes...
    for env in DIVERSE_10:
        for run in [1]: # just one run for the moment...

            # see if we can get full curve learning working on pong
            add_job(
                # the idea here is that a small dc will allow pong to train
                # the replay probably needed, and I can remove it if we want, but it might also help distillation
                # training
                f"E11_PerGameGamma (additional)",
                env_name=env,
                run_name=f"game={env} replay_full (seed={run})",
                default_params=replay_full_args,
                epochs=50,
                priority=100,
                seed=run,
                hostname='',
            )

            add_job(
                f"E11_PerGameGamma",
                env_name=env,
                run_name=f"game={env} tvf_10k (seed={run})",
                use_tvf=True,
                gamma=0.9999,
                tvf_gamma=0.9999,
                default_params=standard_args,
                epochs=50,
                priority=-100,
                seed=run,
                hostname='',
            )
            # add_job(
            #     f"E1_1_PerGameGamma",
            #     env_name=env,
            #     run_name=f"game={env} tvf_1k (seed={run})",
            #     use_tvf=True,
            #     gamma=0.999,
            #     tvf_gamma=0.999,
            #     default_params=standard_args,
            #     epochs=50,
            #     priority=50,
            #     seed=run,
            #     hostname='',
            # )
            # add_job(
            #     f"E1_1_PerGameGamma",
            #     env_name=env,
            #     run_name=f"game={env} tvf_30k (seed={run})",
            #     use_tvf=True,
            #     gamma=0.99997,
            #     tvf_gamma=0.99997,
            #     default_params=standard_args,
            #     epochs=50,
            #     priority=50,
            #     seed=run,
            #     hostname='',
            # )
            add_job(
                f"E11_PerGameGamma",
                env_name=env,
                run_name=f"game={env} tvf_s30k (seed={run})",
                use_tvf=True,
                gamma=0.99997,
                tvf_gamma=0.99997,
                default_params=simple_args,
                epochs=50,
                priority=50,
                seed=run,
                hostname='',
            )

            # add_job(
            #     f"E11_PerGameGamma (additional)",
            #     env_name=env,
            #     run_name=f"game={env} rs_30k (seed={run})",
            #     default_params=replay_simple_args,
            #     epochs=50,
            #     priority=50,
            #     seed=run,
            #     hostname='',
            # )

            add_job(
                f"E11_PerGameGamma (additional2)",
                env_name=env,
                run_name=f"game={env} rs_10k (seed={run})",
                default_params=replay_simple_args,
                tvf_gamma=0.9999,
                gamma=0.9999,
                epochs=50,
                priority=50,
                seed=run,
                hostname='',
            )

            add_job(
                f"E11_PerGameGamma",
                env_name=env,
                run_name=f"game={env} tvf_s10k (seed={run})",
                use_tvf=True,
                gamma=0.9999,
                tvf_gamma=0.9999,
                default_params=simple_args,
                epochs=50,
                priority=50,
                seed=run,
                hostname='ML-Rig',
            )
            add_job(
                f"E11_PerGameGamma",
                env_name=env,
                run_name=f"game={env} tvf_inf (seed={run})",
                use_tvf=True,
                gamma=1.0,
                tvf_gamma=1.0,
                default_params=standard_args,
                epochs=50,
                priority=0,
                seed=run,
                hostname='',
            )

def E03():

    # Expected horizons are 10, 100-1000, 3000-10000, and 10,000
    KEY_4 = ["CrazyClimber", "Zaxxon", "Centipede", "BeamRider"]
    for env in KEY_4:
        for run in [1]:  # just one run for the moment...
            sa_sigma = 0.02
            for sa_mu in [-0.01, 0, 0.01]:
                for strategy in ["sa_return", "sa_reward"]:
                    add_job(
                        f"E31_DynamicGamma (test3)",
                        env_name=env,
                        run_name=f"game={env} gamma {strategy} sa_mu={sa_mu} (seed={run})",
                        auto_gamma="gamma",
                        auto_strategy=strategy,
                        default_params=enhanced_args,
                        tvf_force_ext_value_distil=True,
                        sa_mu=sa_mu,
                        sa_sigma=sa_sigma,
                        epochs=20,
                        priority=0,
                        seed=run,  # this makes sure sa seeds are different.
                        hostname='ML-Rig',
                    )

    # # Changes are: less bias, more seeds
    KEY_4 = ["CrazyClimber", "Zaxxon", "Centipede", "BeamRider"]
    for env in KEY_4:
        for run in [1, 2, 3]:  # just one run for the moment...
            sa_sigma = 0.02
            for sa_mu in [0]:
                for strategy in ["sa_return", "sa_reward"]:
                    add_job(
                        f"E31_DynamicGamma (test4)",
                        env_name=env,
                        run_name=f"game={env} gamma {strategy} sa_mu={sa_mu} (seed={run})",
                        auto_gamma="gamma",
                        auto_strategy=strategy,
                        default_params=simple_args,
                        sa_mu=sa_mu,
                        sa_sigma=sa_sigma,
                        epochs=50,
                        priority=50,
                        seed=run,  # this makes sure sa seeds are different.
                        hostname='',
                    )
    # Changes are: better model, and more variance
    for env in KEY_4:
        for run in [1, 2, 3]:  # just one run for the moment...
            sa_sigma = 0.1
            for sa_mu in [0]:
                for strategy in ["sa_return", "sa_reward"]:
                    add_job(
                        f"E31_DynamicGamma (test5)",
                        env_name=env,
                        run_name=f"game={env} gamma {strategy} sa_mu={sa_mu} (seed={run})",
                        auto_gamma="gamma",
                        auto_strategy=strategy,
                        default_params=replay_simple_args,
                        sa_mu=sa_mu,
                        sa_sigma=sa_sigma,
                        epochs=50,
                        priority=-50,
                        seed=run,  # this makes sure sa seeds are different.
                        hostname='',
                    )


def setup_dynamic_gamma():
    """
    Experiments to see how well TVF works when horizon is modified during training
    """

    # for env in ["Breakout", "CrazyClimber", "Hero", "Krull"]:
    #     add_job(
    #         f"DynamicGamma_{env}",
    #         env_name=env,
    #         run_name=f"reference",
    #         default_params=standard_args,
    #         epochs=30,
    #         priority=0,
    #         hostname='',
    #     )
    #     for auto_gamma in ["gamma", "both"]:
    #         for strategy in ["agent_age_slow", "episode_length", "sa"]:
    #             add_job(
    #                 f"DynamicGamma_{env}",
    #                 env_name=env,
    #                 run_name=f"strategy={strategy} auto_gamma={auto_gamma}",
    #                 auto_gamma=auto_gamma,
    #                 auto_strategy=strategy,
    #                 default_params=standard_args,
    #                 epochs=20,
    #                 priority=10 if strategy == "sa" else 0,
    #                 hostname='',
    #             )

    # v2 fixes the sa reset bug, also included sa_return
    counter = 0
    for env in ["Breakout", "CrazyClimber", "Hero", "Krull"]:
        add_job(
            f"DynamicGamma_v2_{env}",
            env_name=env,
            run_name=f"reference",
            default_params=standard_args,
            epochs=30,
            priority=0,
            hostname='',
        )
        for auto_gamma in ["gamma", "both"]:
            for strategy in ["agent_age_slow", "episode_length", "sa_return", "sa_reward"]:
                add_job(
                    f"DynamicGamma_v2_{env}",
                    env_name=env,
                    run_name=f"strategy={strategy} auto_gamma={auto_gamma}",
                    auto_gamma=auto_gamma,
                    auto_strategy=strategy,
                    default_params=standard_args,
                    epochs=20,
                    priority=10 if strategy[:2] == "sa" else 0,
                    seed=counter, # this makes sure sa seeds are different.
                    hostname='',
                )
                counter += 1

    # check if dynamic gamma works with PPO and DNA?
    counter = 0
    for env in ["Breakout", "CrazyClimber"]:
        for algo in ["PPO", "DNA"]:
            for strategy in ["agent_age_slow", "episode_length", "sa_return", "sa_reward"]:
                add_job(
                    f"DynamicGamma_v3_{env}",
                    env_name=env,
                    run_name=f"strategy={strategy} ({algo})",
                    use_tvf=False,
                    architecture='single' if algo == "PPO" else "dual",
                    auto_gamma="both",
                    auto_strategy=strategy,
                    default_params=standard_args,
                    epochs=50,
                    priority=10,
                    seed = counter, # this makes sure sa seeds are different.
                    hostname='',
                )
                counter += 1


    # multiple runs to see if sa is consistent at finding a gamma
    counter = 999
    for env in ["Breakout", "CrazyClimber", "Hero", "Krull"]:
        for run in [1, 2, 3]:
            for auto_gamma in ["gamma", "both"]:
                for strategy in ["sa_return", "sa_reward"]:
                    add_job(
                        f"DynamicGamma_v2_{env}",
                        env_name=env,
                        run_name=f"strategy={strategy} auto_gamma={auto_gamma} run={run}",
                        auto_gamma=auto_gamma,
                        auto_strategy=strategy,
                        default_params=standard_args,
                        epochs=20,
                        priority=0,
                        seed=counter,  # this makes sure sa seeds are different.
                        hostname='',
                    )
                    counter += 1


def setup_ED():
    """
    Episodic discounting experiments
    """

    default_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'architecture': 'dual',
        'export_video': True,
        'epochs': 50,  # really want to see where these go...
        'use_compression': 'auto',
        'warmup_period': 1000,  # helps on some games to make sure they are really out of sync at the beginning of
        'disable_ev': False,    # training.
        'seed': 0,

        # env parameters
        'time_aware': True,
        'terminal_on_loss_of_life': True,  # to match rainbow
        'reward_clipping': 1,  # to match rainbow

        # parameters found by hyperparameter search...
        'max_grad_norm': 5.0,
        'agents': 128,
        'n_steps': 128,
        'policy_mini_batch_size': 512,
        'value_mini_batch_size': 512,
        'policy_epochs': 3,
        'value_epochs': 2,
        'distill_epochs': 1,
        'distill_beta': 1.0,
        'target_kl': -1,
        'ppo_epsilon': 0.1,
        'policy_lr': 2.5e-4,
        'value_lr': 2.5e-4,
        'distill_lr': 2.5e-4,
        'entropy_bonus': 1e-3,
        'tvf_force_ext_value_distill': False,
        'hidden_units': 256,
        'value_transform': 'sqrt',
        'gae_lambda': 0.95, # would be nice to try 0.99...

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
        'tvf_n_step': 80, # makes no difference...
        'tvf_exp_gamma': 2.0, # 2.0 would be faster, but 1.5 tested slightly better.
        'tvf_coef': 0.5,
        'tvf_soft_anchor': 0,
        'tvf_exp_mode': "transformed",

        # horizon
        'gamma': 1.0,
        'tvf_gamma': 1.0,
        'tvf_max_horizon': 30000,
    }

    # we want 5 interesting games (atari5?) and at least one of each of the episodic discounting.

    for env in DIVERSE_5:

        # reference TVF run, but with no discounting
        add_job(
            f"ED_{env}",
            env_name=env,
            run_name="reference (inf)",
            default_params=default_args,
            priority=-100,
            hostname='ML-Rig',
        )

        # reference TVF run, with very small discounting
        add_job(
            f"ED_{env}",
            env_name=env,
            run_name="reference (30k)",
            default_params=default_args,
            gamma=0.99997,
            tvf_gamma=0.99997,
            priority=-100,
            hostname='ML-Rig',
        )

        for ed_type in ["finite", "geometric", "quadratic", "power", "none"]:
            ed_gamma = 0.99997
            add_job(
                f"ED_{env}",
                env_name=env,
                run_name=f"ed_type={ed_type} ed_gamma={ed_gamma}",
                default_params=default_args,
                ed_type = ed_type,
                ed_gamma = ed_gamma,
                priority=-100,
                hostname='ML-Rig',
            )

def setup_TVF_Atari57():

    """ setup experiment for a few extra ideas."""
    default_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'architecture': 'dual',
        'export_video': True,
        'epochs': 50,  # really want to see where these go...
        'use_compression': 'auto',
        'warmup_period': 1000,  # helps on some games to make sure they are really out of sync at the beginning of
        'disable_ev': False,    # training.
        'seed': 0,

        # env parameters
        'time_aware': False,
        'terminal_on_loss_of_life': True,  # to match rainbow
        'reward_clipping': 1,  # to match rainbow

        # parameters found by hyperparameter search...
        'max_grad_norm': 5.0,
        'agents': 512,
        'n_steps': 128,
        'policy_mini_batch_size': 512,
        'value_mini_batch_size': 1024,
        'policy_epochs': 3,
        'value_epochs': 3,
        'distill_epochs': 3,
        'distill_beta': 1.0,
        'target_kl': -1,
        'ppo_epsilon': 0.3,
        'policy_lr': 2.5e-4,
        'value_lr': 1e-4,
        'distill_lr': 1e-4,
        'entropy_bonus': 1e-3,
        'tvf_force_ext_value_distill': False,
        'hidden_units': 256,
        'value_transform': 'sqrt',
        'gae_lambda': 0.97, # would be nice to try 0.99...

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
        'tvf_n_step': 80, # makes no difference...
        'tvf_exp_gamma': 1.5, # 2.0 would be faster, but 1.5 tested slightly better.
        'tvf_coef': 0.5,
        'tvf_soft_anchor': 0,
        'tvf_exp_mode': "transformed",

        # horizon
        'gamma': 0.99997,
        'tvf_gamma': 0.99997,
        'tvf_max_horizon': 30000,
    }

    for env in canonical_57:
        if env in ["Surround"]:
            continue

        add_job(
            f"TVF_Atari57",
            env_name=env,
            run_name=f"{env}_best",
            default_params=default_args,
            priority=50,
            hostname='ML-Rig',
        )



def test_distil():

    default_distil_args = enhanced_args.copy()
    default_distil_args["use_tvf"] = True
    default_distil_args["gamma"] = 0.99997
    default_distil_args["tvf_gamma"] = 0.99997
    default_distil_args["use_compression"] = False
    default_distil_args["tvf_force_ext_value_distill"] = True
    default_distil_args["epochs"] = 20
    default_distil_args["seed"] = 1
    default_distil_args["hostname"] = ''
    default_distil_args["use_mutex"] = True

    ROLLOUT_SIZE = 128 * 128

    # let's just do all combinations and see what happens
    for env in ['Pong', 'Breakout', 'CrazyClimber']:
        add_job(
            f"test_distil_rp",
            env_name=env,
            run_name=f"game={env} reference (single)",
            architecture='single',
            default_params=default_distil_args,
            priority=40 + 5 if env == "Pong" else 0,
        )
        add_job(
            f"test_distil_rp",
            env_name=env,
            run_name=f"game={env} reference (dual)",
            architecture='dual',
            default_params=default_distil_args,
            priority=40 + 5 if env == "Pong" else 0,
        )
        add_job(
            f"test_distil_rp",
            env_name=env,
            run_name=f"game={env} reference (ppo)",
            use_tvf=False,
            architecture='single',
            default_params=default_distil_args,
            priority=40 + 5 if env == "Pong" else 0,
        )
        for replay_size in [0, ROLLOUT_SIZE, ROLLOUT_SIZE * 2]:
            for replay_mode in ['overwrite', 'uniform', 'mixed']:
                if replay_size == 0 and replay_mode != 'overwrite':
                    continue
                for period in [1, 2]:
                    for epochs in [1, 2]:
                        for dc in [True, False]:
                            if replay_size == 0:
                                rp_tag = ' RP0'
                            else:
                                rp_tag = f" RP{replay_size//(ROLLOUT_SIZE)} ({replay_mode})"

                            add_job(
                                f"test_distil_rp",
                                env_name=env,
                                run_name=f"game={env} {epochs}{period}{rp_tag}{' DC' if dc else ''}",
                                replay_size=replay_size,
                                replay_mode="uniform" if replay_mode is "mixed" else replay_mode,
                                replay_mixing=replay_mode == "mixed",
                                dna_dual_constraint=1.0 if dc else 0.0,
                                distil_period=period,
                                distill_epochs=epochs,
                                default_params=default_distil_args,
                                priority=0 + 5 if env == "Pong" else 0,
                            )

        add_job(
            f"test_distil_rp_long",
            env_name=env,
            run_name=f"game={env} replay_simple mode=off rs={0}",
            replay_size=0,
            distil_batch_mode="full",
            distil_period=1,
            default_params=replay_simple_args,
            priority=40,
            hostname="ML-Rig",
        )

        add_job(
            # this is a really helpful reference run
            f"test_distil_rp_long",
            env_name=env,
            run_name=f"game={env} rp_121 rs={0}",

            replay_size=1 * ROLLOUT_SIZE,
            distil_period=2,
            distil_epochs=1,

            distil_batch_mode="full",
            default_params=replay_simple_args,
            priority=40,
            epochs=20 if env == "Pong" else 50,
            hostname="ML-Rig",
        )

        add_job(
            # just testing a new idea
            f"test_distil_rp_long (v2)",
            env_name=env,
            run_name=f"game={env} rp_h11 rs={0}",

            distil_batch_size=ROLLOUT_SIZE // 2,
            distil_period=1,
            distil_epochs=1,

            default_params=replay_simple_args,
            priority=600,
            epochs=20 if env == "Pong" else 50,
            hostname="ML-Rig",
        )

        add_job(
            # just testing a new idea
            f"test_distil_rp_long",
            env_name=env,
            run_name=f"game={env} rp_s11 rs={0}",

            replay_size=ROLLOUT_SIZE,
            distil_period=1,
            distil_epochs=1,

            distil_lr=1e-4,

            distil_batch_mode="sample",
            default_params=replay_simple_args,
            priority=250,
            epochs=20 if env == "Pong" else 50,
            hostname="ML-Rig",
        )

        def get_rp_epochs(env, mode, replay_size):
            if mode == "sample" and replay_size in [0, 1]:
                # sample 1 and full 1 are the same so ignore sample 1
                return 0
            if env == "Pong":
                return 20
            return 50

        for replay_size in [1, 2, 4, 8]:
            # this has insane memory requirements (because we haven't implemented compression yet)
            # so I need to make sure they go on the ML rig, and happen one at a time.
            add_job(
                f"test_distil_rp_long",
                env_name=env,
                run_name=f"game={env} replay_simple mode=full rs={replay_size}",
                replay_size=replay_size * ROLLOUT_SIZE,
                distil_batch_mode="full",
                distil_period=replay_size,
                default_params=replay_simple_args,
                epochs=get_rp_epochs(env, "full", replay_size),
                priority=40,
                hostname="ML-Rig",
            )
            # this is the alternative method where we just sample
            add_job(
                f"test_distil_rp_long",
                env_name=env,
                run_name=f"game={env} replay_simple mode=sample rs={replay_size}",
                replay_size=replay_size * ROLLOUT_SIZE,
                distil_batch_mode="sample",
                distil_period=1,
                default_params=replay_simple_args,
                epochs=get_rp_epochs(env, "sample", replay_size),
                priority=40,  # just want to start some of these big ones early...
                hostname="ML-Rig",
          )

    for env in ['Pong', 'Breakout', 'CrazyClimber']:

        # find a good dual constraint
        replay_mode = "uniform"
        for replay_size in [0]:
            for period in [1]:
                for epochs in [1]:
                    for dc in [0, 0.1, 0.3, 1.0, 3.0]:
                        if replay_size == 0:
                            rp_tag = ' RP0'
                        else:
                            rp_tag = f" RP{replay_size//(ROLLOUT_SIZE)} ({replay_mode})"
                        add_job(
                            f"test_distil_dc",
                            env_name=env,
                            run_name=f"game={env} {epochs}{period}{rp_tag} DC={dc}",
                            replay_size=replay_size,
                            replay_mode="uniform" if replay_mode is "mixed" else replay_mode,
                            replay_mixing=replay_mode == "mixed",
                            dna_dual_constraint=dc,
                            distil_period=period,
                            distill_epochs=epochs,
                            default_params=default_distil_args,
                            priority=20 + (5 if env == "Pong" else 0),
                        )

E01()
E03()
test_distil()
