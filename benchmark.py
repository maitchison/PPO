"""
Benchmarking for PPO.
"""
import os
import numpy as np
import torch.cuda
import socket
import sys
import argparse
from rl.config import str2bool
from rl.utils import Color

import runner_tools
import plot_util as pu

# todo:
# check how few epochs we can get away with...
#

ROLLOUT_SIZE = 128*128
WORKERS = 8
GPUS = torch.cuda.device_count()
DEVICE = socket.gethostname()
EXPERIMENT_FOLDER = f"__benchmark_{DEVICE}"
EPOCHS = 4 * ROLLOUT_SIZE / 1e6 # (4 would probably also be ok...)

benchmark_args = {
    'checkpoint_every': 0,
    'workers': WORKERS,
    'architecture': 'dual',
    'export_video': False,
    'epochs': EPOCHS,
    'use_compression': True,
    'warmup_period': 0,
    'seed': 0,
    'mutex_key': "DEVICE",
    'benchmark_mode': True,

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
    'max_micro_batch_size': 512,
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
    job_params = benchmark_args.copy()
    job_params.update(params)
    job = runner_tools.Job(EXPERIMENT_FOLDER, run_name, params=job_params)
    p = job.run(run_async=True, force_no_restore=True, verbose=verbose)
    return p


def generate_benchmark_result(parallel_jobs=0, show_compression_stats = False, **params):
    """
    Runs pong for 10M three times and checks the results.

    parallel_jobs = 0 runs single task and returns scalar

    """
    job_results = []

    if parallel_jobs == 0:
        is_scalar = True
        parallel_jobs = 1
    else:
        is_scalar = False

    for seed in range(parallel_jobs):
        job_name = f"pong_{seed}"
        p = execute_job(
            job_name,
            seed=seed,
            env_name="Pong",
            device=f'cuda:{seed % GPUS}',
            use_compression=args.use_compression,
            observation_normalization=args.observation_normalization,
            quiet_mode=not args.verbose,
            **params
        )
        job_results.append((job_name, p))

    results = []

    for job_name, job_result in job_results:
        # wait for job to finish... will take some time...
        outs, errs = job_result.communicate()
        output = outs.decode("utf-8")

        had_error = False

        for line in output.split("\n"):
            if show_compression_stats and line.startswith("Compression stats:"):
                print(line)
            if line.startswith("IPS: "):
                ips_part = line.split(" ")[1]
                ips_part = ips_part.replace(',', '')
                results.append(int(ips_part))
                break
        else:
            had_error = True
            results.append(None)
        if args.verbose or had_error:
            for line in outs.decode("utf-8").split("\n"):
                print(Color.OKGREEN+line+Color.ENDC)
            for line in errs.decode("utf-8").split("\n"):
                print(Color.FAIL+line+Color.ENDC)

    return results[0] if is_scalar else results

def benchmark_quick():

    # clean start
    os.system(f'rm -r ./Run/{EXPERIMENT_FOLDER}')

    ips_results = []

    print(f"Running quick benchmark on {DEVICE} with {GPUS} GPUS...")

    ips = generate_benchmark_result(show_compression_stats=True)
    print(f"Quick result (single-job): {ips:,}")


def benchmark_full():

    # clean start
    os.system(f'rm -r ./Run/{EXPERIMENT_FOLDER}')

    print(f"Running full benchmark on {DEVICE} with {GPUS} GPUS...")

    job_counts = [1, 1 * GPUS, 2 * GPUS]
    baseline_ips = None

    for jobs in job_counts:
        ips = sum(generate_benchmark_result(parallel_jobs=jobs))
        if baseline_ips is None:
            baseline_ips = ips
        ratio = ips / baseline_ips
        print(f"{jobs}-job: {round(ips):,} {ratio:.1f}x")


if __name__ == "__main__":
    orig_stdout = sys.stdout
    orig_stdin = sys.stdin
    try:

        # see https://github.com/pytorch/pytorch/issues/37377 :(
        os.environ["MKL_THREADING_LAYER"] = "GNU"

        parser = argparse.ArgumentParser(description="Trainer for PPO2")
        parser.add_argument("mode", type=str)
        parser.add_argument("--verbose", type=str2bool, default=False)
        parser.add_argument("--use_compression", type=str2bool, default=True)
        parser.add_argument("--observation_normalization", type=str2bool, default=True)

        args = parser.parse_args()

        mode = args.mode.lower()
        if mode == "full":
            benchmark_full()
        elif mode == "quick":
            benchmark_quick()
        else:
            raise Exception(f"Invalid mode {args.mode}")

    finally:
        # not sure why I have to manually reset this back?
        sys.stdout = orig_stdout
        sys.stdin = orig_stdin
