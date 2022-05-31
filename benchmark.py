"""
Benchmarking for PPO.
"""
import os
import numpy as np
import ast
import torch.cuda
import socket
import sys
import argparse
from rl.config import str2bool
from rl.utils import Color
from rl import utils

import runner_tools
import plot_util as pu

# todo:
# check how few epochs we can get away with...
#

ROLLOUT_SIZE = 128*128
WORKERS = 8
GPUS = torch.cuda.device_count()
DEVICE = socket.gethostname()
BENCHMARK_FOLDER = f"__benchmark_{DEVICE}"
REGRESSION_FOLDER = f"__regression_{DEVICE}"
NUMA_GROUPS = utils.detect_numa_groups()
BENCHMARK_EPOCHS = 4 * ROLLOUT_SIZE / 1e6 # (4 should probably be enough)
BENCHMARK_ENV = 'Zaxxon' # a bit of a better test for compression.

regression_args = {
    'checkpoint_every': 0,
    'workers': WORKERS,
    'architecture': 'dual',
    'export_video': False,
    'epochs': 10,

    'warmup_period': 0,
    'seed': 0,
    'mutex_key': "DEVICE",
    'benchmark_mode': True,

    'upload_batch': True,
    'use_compression': False,

    # env parameters
    'embed_time': True,
    'terminal_on_loss_of_life': False,
    'reward_clipping': "off",
    'value_transform': 'identity',

    # parameters found by hyperparameter search...
    'max_grad_norm': 5.0,
    'agents': 128,
    'n_steps': 128,
    'policy_mini_batch_size': 2048,
    'value_mini_batch_size': 512,
    'distil_mini_batch_size': 512,
    'max_micro_batch_size': 512,
    'policy_epochs': 2,
    'value_epochs': 1,

    'target_kl': -1,
    'ppo_epsilon': 0.2,
    'policy_lr': 2.5e-4,
    'value_lr': 2.5e-4,
    'distil_lr': 2.5e-4,
    'entropy_bonus': 1e-3,
    'tvf_force_ext_value_distil': False,
    'hidden_units': 512,
    'gae_lambda': 0.95,

    # new params
    'observation_normalization': True,

    # distil params
    'distil_epochs': 2,
    'distil_beta': 1.0,

    # tvf params
    'use_tvf': True,
    'tvf_value_distribution': 'fixed_geometric',
    'tvf_horizon_distribution': 'fixed_geometric',
    'tvf_horizon_scale': 'log',
    'tvf_time_scale': 'log',
    'tvf_hidden_units': 256,
    'tvf_value_samples': 128,
    'tvf_horizon_samples': 128,
    'tvf_return_mode': 'exponential',
    'tvf_coef': 0.5,
    'tvf_soft_anchor': 0,
    'tvf_exp_mode': "transformed",

    # horizon
    'gamma': 0.99997,
    'tvf_gamma': 0.99997,
    'tvf_max_horizon': 30000,
}


def execute_job(folder:str, run_name: str, numa_id: int = None, **params):
    job_params = regression_args.copy()
    job_params.update(params)
    job = runner_tools.Job(folder, run_name, params=job_params)

    if numa_id is not None:
        # our machine has broken numa where the nodes are 0,2 instead of 0,1,2,3
        # I get around this by manually mapping numa nodes and memory nodes
        # ideally I would juse use
        # preamble = f"numactl --cpunodebind={numa_id} --preferred={numa_id}"
        cpu_mapping = {
            0: '0-5,24-29',
            1: '12-17,36-41',
            2: '6-11,30-35',
            3: '18-23,42-47',
        }
        memory_mapping = {
            0: 0,
            1: 0,
            2: 2,
            3: 2,
        }
        preamble = f"numactl --physcpubind=\"{cpu_mapping[numa_id]}\" --preferred={memory_mapping[numa_id]}"
    else:
        preamble = ""

    p = job.run(run_async=True, force_no_restore=True, preamble=preamble, verbose=args.verbose)
    return p

def print_outputs(outs, errs):
    """
    Prints to screen the output from a popen call.
    """
    for line in outs.decode("utf-8").split("\n"):
        print(Color.OKGREEN + line + Color.ENDC)
    for line in errs.decode("utf-8").split("\n"):
        print(Color.FAIL + line + Color.ENDC)


def generate_benchmark_result(parallel_jobs=0, show_compression_stats=False, device='AUTO', **params):
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
        job_name = f"{BENCHMARK_ENV}_{seed}"

        # set default parameters
        params["use_compression"] = params.get("use_compression", args.use_compression)
        params["observation_normalization"] = params.get("observation_normalization", args.observation_normalization)

        if "env_name" not in params:
            params['env_name'] = BENCHMARK_ENV

        p = execute_job(
            folder=BENCHMARK_FOLDER,
            run_name=job_name,
            seed=seed,
            device=f'cuda:{seed % GPUS}' if device == 'AUTO' else device,
            numa_id=args.numa[seed % len(args.numa)] if args.numa is not None else None,
            epochs=BENCHMARK_EPOCHS,
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
            print_outputs(outs, errs)

    return results[0] if is_scalar else results

def run_suite():
    """
    Runs a suite of tests on a specific GPU
    """

    import os

    # for device in ['cuda:0', 'cuda:1']:
    #     for mbs in [128, 256, 512, 1024]:
    #         custom_args = {
    #             'device': device,
    #             'max_micro_batch_size': mbs,
    #         }
    #         cmd = f'python benchmark.py quick --parallel_jobs=2 --use_compression=False ' \
    #               f'--custom_args="{custom_args}" --verbose=False'
    #         print(cmd)
    #         os.system(cmd)

    # for device in ['cuda:0', 'cuda:1']:
    #     for use_compression in [True, False]:
    #         print("-"*60)
    #         print(f" device:{device:<10} compression:{use_compression:<10}")
    #         print("-" * 60)
    #
    #         custom_args = {
    #             'device': device,
    #         }
    #
    #         for jobs in [1, 2, 3, 4]:
    #             cmd = f'python benchmark.py quick --parallel_jobs={jobs} --use_compression={use_compression} ' \
    #                   f'--custom_args="{custom_args}" --verbose=False'
    #             print(cmd)
    #             os.system(cmd)

    for device in ['cuda:0', 'cuda:1']:
        for mutex_key in ['']:

            custom_args = {
                'device': device,
                'mutex_key': mutex_key,
            }

            for jobs in [1, 2, 3]:
                cmd = f'python benchmark.py quick --parallel_jobs={jobs} --use_compression=False ' \
                      f'--custom_args="{custom_args}" --verbose=False'
                print(cmd)
                os.system(cmd)


def run_benchmark(description: str, job_counts: list, **kwargs):

    # clean start
    os.system(f'rm -r ./Run/{BENCHMARK_FOLDER}')

    if args.verbose:
        print(f"Running {description} benchmark on {DEVICE} with {GPUS} GPUS...")

    baseline_ips = None

    if args.custom_args is not None:
        kwargs = kwargs.copy()
        kwargs.update(ast.literal_eval(args.custom_args))

    for jobs in job_counts:
        ips = sum(
            generate_benchmark_result(parallel_jobs=jobs, show_compression_stats=jobs == job_counts[0], **kwargs)
        )
        if baseline_ips is None:
            baseline_ips = ips
        ratio = ips / baseline_ips
        print(f"{jobs}-job: {round(ips):,} {ratio:.1f}x")


def run_regressions():
    """
    Runs pong for 10M three times and checks the results.
    """
    job_results = []

    print("Running regression.")

    for seed in [0, 1, 2, 3]:
        job_name = f"pong_{seed}"
        p = execute_job(
            folder=REGRESSION_FOLDER,
            run_name=job_name,
            env_name="Pong",
            seed=seed,
            device=f'cuda:{seed % GPUS}',
            quiet_mode=not args.verbose
        )
        job_results.append((job_name, p))

    for job_name, job_result in job_results:
        # wait for job to finish... will take some time...
        outs, errs = job_result.communicate()
        if args.verbose:
            print_outputs(outs, errs)

    show_regression_results()

def show_regression_results():
    """ Loads latest regression result and prints outout. """
    mean_scores = []
    aoc_scores = []
    for seed in [0, 1, 2, 3]:
        job_name = f"pong_{seed}"
        logs = pu.get_runs("./Run/"+REGRESSION_FOLDER, run_filter=lambda x : job_name in x)
        assert len(logs) == 1, f"Expected one log with name '{job_name}' " \
                               f"in folder {REGRESSION_FOLDER} but found {len(logs)}."
        log = logs[0]
        scores = log[1]["ep_score_mean"]
        mean_scores.append(np.mean(scores[-50:]))  # from last sixth
        aoc_scores.append(np.mean(scores))  # over entire curve

    min_score = np.min(mean_scores)
    max_score = np.max(mean_scores)
    avg_score = np.mean(mean_scores)
    aoc_score = np.mean(aoc_scores)
    print(f"Result: {Color.WARNING}{avg_score:.1f}{Color.ENDC} ({min_score:.1f}-{max_score:.1f})"
          f" [aoc={aoc_score:.1f}]")
    did_pass = min_score > 15.0 and avg_score > 20.0
    if did_pass:
        print(f"{Color.OKGREEN}[PASS]{Color.ENDC}")
    else:
        print(f"{Color.FAIL}[FAIL]{Color.ENDC}")


if __name__ == "__main__":
    orig_stdout = sys.stdout
    orig_stdin = sys.stdin
    try:

        # numa doesn't help...
        # if NUMA_GROUPS is not None:
        #     print(f"Detected {NUMA_GROUPS} NUMA nodes.")

        # see https://github.com/pytorch/pytorch/issues/37377 :(
        os.environ["MKL_THREADING_LAYER"] = "GNU"

        parser = argparse.ArgumentParser(description="Benchmarker")
        parser.add_argument("mode", type=str, help="[full|quick|ppo|regression|show|suite]")
        parser.add_argument("--verbose", type=str2bool, default=False)
        parser.add_argument("--parallel_jobs", type=int, default=1)
        parser.add_argument("--use_compression", type=str2bool, default=False)
        parser.add_argument("--observation_normalization", type=str2bool, default=True)
        parser.add_argument("--numa", type=str, default=None, help='e.g. [0,1]')
        parser.add_argument("--jobs", type=str, default=None)
        parser.add_argument("--custom_args", type=str, default=None)

        args = parser.parse_args()

        if args.numa is not None:
            args.numa = ast.literal_eval(args.numa)
            print(f"Using numa: {args.numa})")

        mode = args.mode.lower()
        if mode == "full":
            if args.jobs is not None:
                job_override = ast.literal_eval(args.jobs)
                run_benchmark("custom", job_override)
            else:
                run_benchmark("full", [1, 1 * GPUS, 2 * GPUS])
        elif mode == "quick":
            run_benchmark("quick", [args.parallel_jobs])
        elif mode == "ppo":
            run_benchmark(
                "ppo",
                [1, 1 * GPUS, 2 * GPUS, 3 * GPUS],
                use_tvf=False,
                architecture="single",
            )
        elif mode == "regression":
            run_regressions()
        elif mode == "suite":
            run_suite()
        elif mode == "show":
            show_regression_results()
        else:
            raise Exception(f"Invalid mode {args.mode}")

    finally:
        # not sure why I have to manually reset this back?
        sys.stdout = orig_stdout
        sys.stdin = orig_stdin
