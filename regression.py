"""
Regression test for training.

A number of seeded experiments are run and the results are compared against.

Running the experiment requires 20M epochs, which at 4x1000 IPS should take around 1-hour.

"""
import os

import runner_tools
import plot_util as pu
import torch
import socket
import shutil

ROLLOUT_SIZE = 128*128
WORKERS = 8
GPUS = torch.cuda.device_count()
DEVICE = socket.gethostname()
EXPERIMENT_FOLDER = f"__regression_{DEVICE}"
EPOCHS = 4 * ROLLOUT_SIZE / 1e6 # (4 would probably also be ok...)

DEBUG = False

regression_args = {
    'checkpoint_every': 0,
    'workers': WORKERS,
    'architecture': 'dual',
    'export_video': False,
    'epochs': 5,  # just enough to make sure it's working.
    'use_compression': 'auto',
    'warmup_period': 1000,
    'seed': 0,
    'mutex_key': "DEVICE",

    # env parameters
    'time_aware': True,
    'terminal_on_loss_of_life': False,
    'reward_clipping': "off",
    'value_transform': 'identity',

    # parameters found by hyperparameter search...
    'max_grad_norm': 5.0,
    'agents': 128,
    'n_steps': 128,
    'policy_mini_batch_size': 512,
    'value_mini_batch_size': 512,
    'policy_epochs': 3,
    'value_epochs': 2,

    'target_kl': -1,
    'ppo_epsilon': 0.1,
    'policy_lr': 2.5e-4,
    'value_lr': 2.5e-4,
    'distill_lr': 2.5e-4,
    'entropy_bonus': 1e-3,
    'tvf_force_ext_value_distill': False,
    'hidden_units': 256,
    'gae_lambda': 0.95,

    # new params
    'observation_normalization': True,
    'observation_scaling': "centered",
    'layer_norm': False,

    # distil params
    'distil_epochs': 1,
    'distil_beta': 1.0,

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
    'tvf_n_step': 80,  # makes no difference...
    'tvf_exp_gamma': 2.0,  # 2.0 would be faster, but 1.5 tested slightly better.
    'tvf_coef': 0.5,
    'tvf_soft_anchor': 0,
    'tvf_exp_mode': "transformed",

    # horizon
    'gamma': 0.99997,
    'tvf_gamma': 0.99997,
    'tvf_max_horizon': 30000,
}


def execute_job(run_name: str, verbose=False, **params):
    job_params = regression_args.copy()
    job_params.update(params)
    job = runner_tools.Job(EXPERIMENT_FOLDER, run_name, params=job_params)
    p = job.run(run_async=True, force_no_restore=True, verbose=verbose)
    return p


def run_regressions():
    """
    Runs pong for 10M three times and checks the results.
    """
    job_results = []

    print("Running regression.")

    for seed in [0, 1, 2]:
        job_name = f"pong_{seed}"
        p = execute_job(job_name, seed=seed, env_name="Pong", device=f'cuda:{seed % GPUS}', verbose=DEBUG)
        job_results.append((job_name, p))

    for job_name, job_result in job_results:
        # wait for job to finish... will take some time...
        outs, errs = job_result.communicate()
        if DEBUG:
            print(outs)
            print(errs)

    print("Done.")

def check_results():
    for seed in [0, 1, 2, 3]:
        job_name = f"pong_{seed}"
        log = pu.get_runs(f"./Run/{EXPERIMENT_FOLDER}", run_filter=lambda x: job_name)
        assert len(log) == 1
        log = log[0]

        # check score
        # check ev?
        # check losses / grads ?


def main():

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    # clean start
    shutil.rmtree(f'./Run/{EXPERIMENT_FOLDER}')
    run_regressions()
    #check_results()


if __name__ == "__main__":
    main()


