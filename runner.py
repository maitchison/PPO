import numpy
import os
import sys
import pandas as pd
import json
import time
import random
import numpy as np

from rl import utils

import socket
HOST_NAME = socket.gethostname()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

CHUNK_SIZE = 10
DEVICE = "auto"
OUTPUT_FOLDER = "./Run"
WORKERS = 4

if len(sys.argv) == 3:
    DEVICE = sys.argv[2]

def add_job(experiment_name, run_name, priority=0, chunked=True, **kwargs):

    if "device" not in kwargs:
        kwargs["device"] = DEVICE

    job_list.append(Job(experiment_name, run_name, priority, chunked, kwargs))

def get_run_folder(experiment_name, run_name):
    """ Returns the path for given experiment and run, or none if not found. """

    path = os.path.join(OUTPUT_FOLDER, experiment_name)
    if not os.path.exists(path):
        return None

    for file in os.listdir(path):
        name = os.path.split(file)[-1]
        this_run_name = name[:-(8+3)]  # crop off the id code.
        if this_run_name == run_name:
            return os.path.join(path, name)
    return None

class Job:

    """
    Note: we don't cache any of the properties here as other worker may modify the filesystem, so we need to always
    use the up-to-date version.
    """

    # class variable to keep track of insertion order.
    id = 0

    def __init__(self, experiment_name, run_name, priority, chunked, params):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.priority = priority
        self.params = params
        self.id = Job.id
        self.chunked = chunked
        Job.id += 1

    def __lt__(self, other):
         return self._sort_key < other._sort_key

    @property
    def _sort_key(self):

        status = self.get_status()

        priority = self.priority

        # make running tasks appear at top...
        if status == "working":
            priority += 1000

        if "search" in self.experiment_name:
            # with search we want to make sure we complete partial runs first
            priority = priority + self.get_completed_epochs()
        else:
            priority = priority - self.get_completed_epochs()

        return (-priority, self.get_completed_epochs(), self.experiment_name, self.id)

    def get_path(self):
        # returns path to this job.
        return get_run_folder(self.experiment_name, self.run_name)

    def get_status(self):

        path = self.get_path()
        if path is None or not os.path.exists(path):
            return "pending"

        status = "pending"

        last_modifed = None

        if os.path.exists(os.path.join(path, "params.txt")):
            status = "waiting"

        details = self.get_details()
        if details is not None and details["fraction_complete"] >= 1.0:
            status = "completed"

        if os.path.exists(os.path.join(path, "lock.txt")):
            last_modifed = os.path.getmtime(os.path.join(path, "lock.txt"))
            status = "working"

        if status in ["working"] and last_modifed is not None:
            hours_since_modified = (time.time()-last_modifed)/60/60
            if hours_since_modified > 1.0:
                status = "stale"

        return status

    def get_data(self):
        try:
            path = os.path.join(self.get_path(), "training_log.csv")
            return pd.read_csv(path)
        except:
            return None

    def get_params(self):
        try:
            path = os.path.join(self.get_path(), "params.txt")
            return json.load(open(path, "r"))
        except:
            return None

    def get_details(self):
        try:
            path = os.path.join(self.get_path(), "progress.txt")

            details = json.load(open(path, "r"))

            # if max_epochs has changed fix up the fraction_complete.
            if details["max_epochs"] != self.params["epochs"]:
                details["fraction_complete"] = details["completed_epochs"] / self.params["epochs"]

            return details
        except:
            return None

    def get_completed_epochs(self):
        details = self.get_details()
        if details is not None:
            return details["completed_epochs"]
        else:
            return 0

    def run(self, chunked=False):

        self.params["output_folder"] = OUTPUT_FOLDER

        experiment_folder = os.path.join(OUTPUT_FOLDER, self.experiment_name)

        # make the destination folder...
        if not os.path.exists(experiment_folder):
            print("Making new experiment folder {experiment_folder}")
            os.makedirs(experiment_folder, exist_ok=True)

        # copy script across if needed.
        train_script_path = utils.copy_source_files("./", experiment_folder)

        self.params["experiment_name"] = self.experiment_name
        self.params["run_name"] = self.run_name

        details = self.get_details()

        if details is not None and details["completed_epochs"] > 0:
            # restore if some work has already been done.
            self.params["restore"] = True
            print(f"Found restore point {self.get_path()} at epoch {details['completed_epochs']}")
        else:
            print(f"No restore point found for path {self.get_path()}")

        if chunked:
            # work out the next block to do
            if details is None:
                next_chunk = CHUNK_SIZE
            else:
                next_chunk = (round(details["completed_epochs"] / CHUNK_SIZE) * CHUNK_SIZE) + CHUNK_SIZE
            self.params["limit_epochs"] = int(next_chunk)


        python_part = "python {} {}".format(train_script_path, self.params["env_name"])

        params_part = " ".join([f"--{k}={nice_format(v)}" for k, v in self.params.items() if k not in ["env_name"] and v is not None])
        params_part_lined = "\n".join([f"--{k}={nice_format(v)}" for k, v in self.params.items() if k not in ["env_name"] and v is not None])

        print()
        print("=" * 120)
        print(bcolors.OKGREEN+self.experiment_name+" "+self.run_name+bcolors.ENDC)
        print("Running " + python_part + "\n" + params_part_lined)
        print("=" * 120)
        print()
        return_code = os.system(python_part + " " + params_part)
        if return_code != 0:
            raise Exception("Error {}.".format(return_code))

def run_next_experiment(filter_jobs=None):

    job_list.sort()

    for job in job_list:
        if filter_jobs is not None and not filter_jobs(job):
            continue
        status = job.get_status()

        if status in ["pending", "waiting"]:

            job.get_params()

            job.run(chunked=job.chunked)
            return

def comma(x):
    if type(x) is int or (type(x) is float and int(x) == x):
        return "{:,}".format(x)
    else:
        return x

def show_experiments(filter_jobs=None, all=False):
    job_list.sort()
    print("-" * 151)
    print("{:^10}{:<20}{:<60}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}".format("priority", "experiment_name", "run_name", "complete", "status", "eta", "fps", "score", "host"))
    print("-" * 151)
    for job in job_list:

        if filter_jobs is not None and not filter_jobs(job):
                continue

        status = job.get_status()

        if status == "completed" and not all:
            continue

        details = job.get_details()

        if details is not None:
            percent_complete = "{:.1f}%".format(details["fraction_complete"]*100)
            eta_hours = "{:.1f}h".format(details["eta"] / 60 / 60)
            score = details["score"]
            if score is None: score = 0
            score = "{:.1f}".format(score)
            host = details["host"][:8]
            fps = int(details["fps"])
        else:
            percent_complete = ""
            eta_hours = ""
            score = ""
            host = ""
            fps = ""

        status_transform = {
            "pending": "",
            "stale": "stale",
            "completed": "done",
            "working": "running",
            "waiting": "pending"
        }

        print("{:^10}{:<20}{:<60}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
            job.priority, job.experiment_name[:19], job.run_name, percent_complete, status_transform[status], eta_hours, comma(fps), comma(score), host))


def setup_experiments6():

    default_args = {
        'env_name': "Breakout",
        'checkpoint_every': int(5e6),
        'epochs': 50,
        'agents': 256,
        'n_steps': 128,
        'max_grad_norm': 5.0,
        'entropy_bonus': 0.01,
        'use_tvf': True,
        'tvf_coef': 0.1,
        'tvf_n_horizons': 128,
        'tvf_advantage': True,
        'vf_coef': 0.0,
        'workers': 8, # we will be running lots of experiments so reduce this down a little... (8 is better though)
        'tvf_gamma': 0.997,
        'gamma': 0.997,
        'n_mini_batches': 32,
    }

    for entropy_bonus in [0.001, 0.003, 0.01, 0.03]:
        add_job(
            "TVF_6A",
            run_name=f"entropy_bonus={entropy_bonus}",
            tvf_max_horizon=1000,
            entropy_bonus=entropy_bonus,
            priority=100,
            **{k: v for k, v in default_args.items() if k != "entropy_bonus"}
        )

    for tvf_max_horizon in [300, 1000, 3000, 10000]:
        add_job(
            "TVF_6B",
            run_name=f"tvf_mh={tvf_max_horizon}",
            tvf_max_horizon=tvf_max_horizon,
            priority=50,
            **default_args
        )

    # as per previous
    for tvf_max_horizon in [999, 1000, 1001, 2000, 4000]:
        add_job(
            "TVF_6B2",
            run_name=f"tvf_mh={tvf_max_horizon}",
            tvf_max_horizon=tvf_max_horizon,
            priority=100,
            **default_args
        )

    for agents in [64, 128, 256, 512]:
        add_job(
            "TVF_6C",
            run_name=f"agents={agents}",
            tvf_max_horizon=1000,
            agents=agents,
            **{k:v for k,v in default_args.items() if k != "agents"}
        )

    for n_steps in [16, 32, 64, 128, 256]:
        add_job(
            "TVF_6D",
            run_name=f"n_steps={n_steps}",
            tvf_max_horizon=1000,
            n_steps=n_steps,
            **{k:v for k,v in default_args.items() if k != "n_steps"}
        )

    for distribution in ["uniform", "linear", "first_and_last"]:
        add_job(
            "TVF_6E",
            run_name=f"tvf_sample_dist={distribution} samples=32",
            tvf_max_horizon=1000,
            tvf_sample_dist=distribution,
            tvf_n_horizons=32,
            **{k:v for k,v in default_args.items() if k != "tvf_n_horizons"}
        )

    for n_mini_batches in [8, 16, 32, 64]:
        add_job(
            "TVF_6F",
            run_name=f"n_mini_batches={n_mini_batches}",
            tvf_max_horizon=1000,
            n_mini_batches=n_mini_batches,
            **{k:v for k,v in default_args.items() if k != "n_mini_batches"}
        )

    for tvf_coef in [0.01, 0.03, 0.1, 0.3, 1]:
        add_job(
            "TVF_6G",
            run_name=f"tvf_coef={tvf_coef}",
            tvf_max_horizon=1000,
            tvf_coef=tvf_coef,
            **{k:v for k,v in default_args.items() if k != "tvf_coef"}
        )

    # make sure new advantage normalization code, and improved sampling works
    for tvf_max_horizon in [1000, 999, 1001]: # just a way to get 3 runs...
        add_job(
            "TVF_6H",
            run_name=f"tvf_mh={tvf_max_horizon}",
            tvf_max_horizon=tvf_max_horizon,
            priority=100,
            **default_args
        )


    # evaluation on MR
    for env in ['MontezumaRevenge']:
        add_job(
            "TVF_6_mr",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=1000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_advantage=True,
            vf_coef=0.0,

            tvf_coef=0.01,
            tvf_n_horizons=64,

            workers=WORKERS,  # we will be running lots of experiments so reduce this down a little... (8 is better though)
            tvf_gamma=0.997,
            gamma=0.997,
            n_mini_batches=32,

            priority=100,
        )

    # evaluation on MR
    for env in ['Alien', 'MontezumaRevenge']:
        add_job(
            "TVF_6_rnd",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=1000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_advantage=True,

            tvf_coef=0.01,
            tvf_n_horizons=64,

            use_rnd=True,
            observation_normalization=True,
            vf_coef=0.5,
            intrinsic_reward_scale=1.0,

            workers=WORKERS,
            # we will be running lots of experiments so reduce this down a little... (8 is better though)
            tvf_gamma=0.997,
            gamma=0.997,
            n_mini_batches=32,

            priority=100,
        )

    # evaluation run (only setting difference is tvf_coef is now 0.01)
    for env in ['Alien', 'BankHeist', 'CrazyClimber']:
        add_job(
            "TVF_6_eval",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=1000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_advantage=True,
            vf_coef=0.0,

            tvf_coef=0.01,
            tvf_n_horizons=64,

            workers=WORKERS,  # we will be running lots of experiments so reduce this down a little... (8 is better though)
            tvf_gamma=0.997,
            gamma=0.997,
            n_mini_batches=32,

            chunked=False,
            priority=100,
        )

    # longer horizon..
    for env in ['Alien', 'BankHeist', 'CrazyClimber']:
        add_job(
            "TVF_6_eval_999",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=3000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_advantage=True,
            vf_coef=0.0,

            tvf_coef=0.01,
            tvf_n_horizons=64,

            workers=WORKERS,
            # we will be running lots of experiments so reduce this down a little... (8 is better though)
            tvf_gamma=0.999,
            gamma=0.999,
            n_mini_batches=32,

            chunked=False,
            priority=60,
        )


    # trying the split model... (so we don't have to balance value and policy...
    for env in ['Alien', 'BankHeist', 'CrazyClimber']:
        add_job(
            "TVF_6_eval_split",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=1000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_advantage=True,
            vf_coef=0.0,

            tvf_coef=0.01,
            tvf_n_horizons=64,
            tvf_model='split',

            workers=WORKERS,  # we will be running lots of experiments so reduce this down a little... (8 is better though)
            tvf_gamma=0.997,
            gamma=0.997,
            n_mini_batches=32,

            chunked=False,
            priority=120,
        )

    # trying the split model... (so we don't have to balance value and policy...
    for env in ['Alien', 'BankHeist', 'CrazyClimber']:
        add_job(
            "TVF_6_eval_joint",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=1000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_advantage=True,
            vf_coef=0.0,

            tvf_coef=0.01,
            tvf_n_horizons=64,
            tvf_model='split',
            tvf_joint_weight=0.1, # just a little hint to make these weights the same

            workers=WORKERS,
            # we will be running lots of experiments so reduce this down a little... (8 is better though)
            tvf_gamma=0.997,
            gamma=0.997,
            n_mini_batches=32,

            priority=175,
        )

    # some RND exploration
    for env in ['Alien', 'BankHeist', 'CrazyClimber']:
        add_job(
            "TVF_6_eval_rnd",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=1000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_advantage=True,

            tvf_coef=0.01,
            tvf_n_horizons=64,

            workers=WORKERS,  # we will be running lots of experiments so reduce this down a little... (8 is better though)
            tvf_gamma=0.997,
            gamma=0.997,
            n_mini_batches=32,

            use_rnd=True,
            observation_normalization=True,
            vf_coef=0.5,
            intrinsic_reward_scale=0.25, # big guess here...

            priority=100,
        )

    # stronger RND
    for env in ['Alien', 'BankHeist', 'CrazyClimber']:
        add_job(
            "TVF_6_eval_rnd2",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=1000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_advantage=True,

            tvf_coef=0.01,
            tvf_n_horizons=64,

            workers=WORKERS,
            # we will be running lots of experiments so reduce this down a little... (8 is better though)
            tvf_gamma=0.997,
            gamma=0.997,
            n_mini_batches=32,

            use_rnd=True,
            observation_normalization=True,
            vf_coef=0.5,
            intrinsic_reward_scale=1.0,  # big guess here...

            priority=100,
        )

    # evaluation run (only setting difference is tvf_coef is now 0.01)
    for env in ['Alien', 'BankHeist', 'CrazyClimber']:
        add_job(
            "TVF_6_eval_high_samples",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=1000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_advantage=True,
            vf_coef=0.0,

            tvf_coef=0.01,
            tvf_n_horizons=256,

            workers=WORKERS,
            tvf_gamma=0.997,
            gamma=0.997,
            n_mini_batches=32,

            priority=100,
        )

    # evaluation run (only setting difference is tvf_coef is now 0.01)
    for env in ['Alien', 'BankHeist', 'CrazyClimber']:
        add_job(
            "TVF_6_eval_99",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=300,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_advantage=True,
            vf_coef=0.0,

            tvf_coef=0.01,
            tvf_n_horizons=64,

            workers=WORKERS,
            # we will be running lots of experiments so reduce this down a little... (8 is better though)
            tvf_gamma=0.99,
            gamma=0.99,
            n_mini_batches=32,

            priority=100,
        )

    for env in ['Alien', 'BankHeist', 'CrazyClimber']:
        add_job(
            "TVF_6_eval_ppo_997",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=1000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=False,
            tvf_advantage=False,
            vf_coef=0.5,

            workers=WORKERS,  # we will be running lots of experiments so reduce this down a little... (8 is better though)
            gamma=0.997,
            n_mini_batches=32,

            priority=85,
        )

    for env in ['Alien', 'BankHeist', 'CrazyClimber']:
        add_job(
            "TVF_6_eval_ppo_999",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=1000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=False,
            tvf_advantage=False,
            vf_coef=0.5,

            workers=WORKERS,  # we will be running lots of experiments so reduce this down a little... (8 is better though)
            gamma=0.999,
            n_mini_batches=32,

            priority=85,
        )

    for env in ['Alien', 'BankHeist', 'CrazyClimber']:
        add_job(
            "TVF_6_eval_ppo_99",
            env_name=env,
            run_name=f"env={env}",
            tvf_max_horizon=1000,

            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,
            n_steps=128,
            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=False,
            tvf_advantage=False,
            vf_coef=0.5,

            workers=WORKERS,  # we will be running lots of experiments so reduce this down a little... (8 is better though)
            gamma=0.99,
            n_mini_batches=32,

            priority=70,
        )


def random_search(run, main_params, search_params, count=128):

    for i in range(count):
        params = {}
        np.random.seed(i)
        for k, v in search_params.items():
            params[k] = np.random.choice(v)

        # make sure params arn't too high (due to memory)
        while params["agents"] * params["n_steps"] > 64*1024:
            params["agents"] //= 2

        add_job(run, run_name=f"{i:04d}", chunked=True, **main_params, **params)


def nice_format(x):
    if type(x) is str:
        return f'"{x}"'
    if x is None:
        return "None"
    if type(x) in [int, float, bool]:
        return str(x)

    return f'"{x}"'

def setup_tvf_random_search():

    main_params = {
        'env_name': "Breakout",
        'checkpoint_every': int(5e6),
        'epochs': 50, # can be done now that n_horizons is low, and we are using fast MC algorithm.
        'use_tvf': True,
        'tvf_advantage': True,
        'vf_coef': 0.0,
        'workers': WORKERS,
        'tvf_gamma': 0.997,
        'gamma': 0.997,
        'priority': -100,
        'tvf_max_horizon': 1000,

    }

    # just want to figure out the interplay between mini_batch_size, n_steps, and n_agents...
    # I should plot 'mini_batch_size' as well... and look for correlations between variables...

    # goal would be to find some good settings, and get 500+ on atari in 50m.

    search_params = {
        'agents': [64, 128, 256, 512],
        'n_steps': [32, 64, 128, 256, 512],
        'tvf_coef': [0.1, 0.03, 0.01, 0.003],
        'n_mini_batches': [4, 8, 16, 32, 64],  # might be better to have this as mini_batch size?

        # I'm just interested to see what effect these have, but I don't expect any interplay.
        'entropy_bonus': [0.03, 0.01, 0.003],
        'ppo_epsilon': [0.1, 0.2], # allow for faster learning
        'tvf_n_horizons': [16, 32, 64], # smaller samples should work and will be faster
        'adam_epsilon': [1e-5, 1e-8],
        'max_grad_norm': [0.5, 5.0, 10.0],
        'tvf_sample_dist': ['uniform', 'linear'],
        'learning_rate': [2.5e-4, 1e-4, 2.5e-5],  # try slower learning rates... might help for 200m
    }

    random_search("tvf_v6_search", main_params, search_params)


def setup_experiments5():

    for tvf_n_step in [16,32,64]:
        add_job(
            "TVF_5B_FIXED",
            run_name=f"tvf_n_step={tvf_n_step}",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,

            agents=64,
            n_steps=512,

            tvf_lambda=-tvf_n_step,

            tvf_gamma=0.997,
            gamma=0.997,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=1000,
            tvf_n_horizons=250,
            tvf_advantage=True,
            tvf_loss_func="nlp",
            tvf_sample_dist="uniform",

            vf_coef=0,
            tvf_epsilon=0.1,

            workers=8,
            time_aware=False,
            priority=200 if tvf_n_step == 32 else 150,
        )

    for tvf_n_step in [8,16,32,64,128,256,512]:
        add_job(
            "TVF_5B_OLD",
            run_name=f"tvf_n_step={tvf_n_step}",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,

            agents=64,
            n_steps=512,

            tvf_lambda=-tvf_n_step,

            tvf_gamma=0.997,
            gamma=0.997,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=1000,
            tvf_n_horizons=250,
            tvf_advantage=True,
            tvf_loss_func="nlp",
            tvf_sample_dist="uniform",

            vf_coef=0,
            tvf_epsilon=0.1,

            workers=8,
            time_aware=False,
            priority=200 if tvf_n_step == 32 else 150,
        )


    for tvf_n_step in [16,32,64]:
        add_job(
            "TVF_5B_SCRATCH",
            run_name=f"tvf_n_step={tvf_n_step}",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,

            agents=64,
            n_steps=512,

            tvf_lambda=-tvf_n_step,

            tvf_gamma=0.997,
            gamma=0.997,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=1000,
            tvf_n_horizons=250,
            tvf_advantage=True,
            tvf_loss_func="nlp",
            tvf_sample_dist="uniform",

            vf_coef=0,
            tvf_epsilon=0.1,

            workers=8,
            time_aware=False,
            priority=200 if tvf_n_step == 32 else 150,
        )



if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    id = 0
    job_list = []
    #setup_mvh4()
    setup_experiments5()
    setup_experiments6()
    setup_tvf_random_search()

    if len(sys.argv) == 1:
        experiment_name = "show"
    else:
        experiment_name = sys.argv[1]

    if experiment_name == "show_all":
        show_experiments(all=True)
    elif experiment_name == "show":
        show_experiments()
    elif experiment_name == "auto":
        run_next_experiment()
    else:
        run_next_experiment(filter_jobs=lambda x: x.experiment_name == experiment_name)