import os
import sys
import json
import time
import random
import platform
import math

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

DEVICE="auto"
OUTPUT_FOLDER="./Run"
WORKERS=4 # saves on memory, workers are about 0.5GB per one, so 8 workers = +4GB per job...

if len(sys.argv) == 3:
    DEVICE = sys.argv[2]


tuned_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'env_name': 'DemonAttack',
        'epochs': 50,
        'max_grad_norm': 0.5,
        'agents': 128,
        'n_steps': 128,
        'policy_mini_batch_size': 512,      # does this imply microbatching is broken?
        'value_mini_batch_size': 512,
        'value_epochs': 2,                  # this seems really low?
        'policy_epochs': 4,
        'target_kl': 0.01,
        'ppo_epsilon': 0.05,
        'value_lr': 5e-4,                   # a bug caused this to be policy_lr...
        'policy_lr': 2.5e-4,
        'gamma': 0.999,
    }

v8_args = {
    'use_tvf': True,
    'tvf_hidden_units': 128,

    'tvf_horizon_samples': 128,
    'tvf_value_samples': 128,

    'tvf_lambda': -16,
    'tvf_coef': 0.01,
    'tvf_max_horizon': 3000,
    'gamma': 0.999,
    'tvf_gamma': 0.999,

    # required due to bug fix
    'value_lr': 2.5e-4,

    'tvf_loss_weighting': "advanced",
    'tvf_h_scale': "squared",

    **tuned_args,
}

hps_14_args = {
    'use_tvf': True,
    'tvf_hidden_units': 128,

    'tvf_horizon_samples': 128,
    'tvf_value_samples': 128,

    'tvf_lambda': -16,
    'tvf_coef': 0.01,
    'tvf_max_horizon': 3000,
    'gamma': 0.999,
    'tvf_gamma': 0.999,

    # required due to bug fix
    'value_lr': 2.5e-4,

    'tvf_loss_weighting': "advanced",
    'tvf_h_scale': "squared",

    **tuned_args,
}


# adjusted for new tvf_coef etc
v14_adjusted = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'epochs': 50,
    'max_grad_norm': 20.0,
    'agents': 256,  # more is probably better
    'n_steps': 128,  # hard to know, but at least we are now free the MC algorithm
    'policy_mini_batch_size': 1024,
    'value_mini_batch_size': 256,  # smaller is probably better
    'value_epochs': 2,  # I probably want 4 with early stopping
    'policy_epochs': 4,
    'target_kl': 0.01,  # need to search more around this
    'ppo_epsilon': 0.1,  # hard to say if this is right...
    'value_lr': 2.5e-4,  # slow and steady wins the race
    'policy_lr': 2.5e-4,
    'export_video': False,  # in general not needed

    'tvf_horizon_distribution': 'advanced',
    'tvf_horizon_scale': 'centered',
    'tvf_update_return_freq': 4,

    'use_tvf': True,
    'tvf_hidden_units': 128,  # big guess here
    'tvf_value_samples': 128,  # reducing from 128 samples to 64 is fine, even 16 would work.
    'tvf_horizon_samples': 64,  # 32 has been shown to be better than 128 (this might have been due to loss scaling)
    'tvf_lambda': -16,
    'tvf_lambda_samples': 32,

    'tvf_coef': 6.4, # due to scaling
    'tvf_soft_anchor': 1.0,

    'tvf_max_horizon': 3000,
    'gamma': 0.999,
    'tvf_gamma': 0.999,
}

# these were set to best settings from search.
v15_args = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'epochs': 50,
    'export_video': False,                  # in general not needed

    # PPO args
    'max_grad_norm': 32.0,
    'agents': 768,
    'n_steps': 128,
    'policy_mini_batch_size': 512,
    'value_mini_batch_size': 512,
    'value_epochs': 2,
    'policy_epochs': 1,
    'distill_epochs': 1,
    'target_kl': 0.15,
    'ppo_epsilon': 0.25,                    # was 0.1
    'value_lr': 2.5e-4,
    'policy_lr': 1.0e-4,
    'entropy_bonus': 0.02,                  # was 0.01
    'time_aware': True,
    'distill_beta': 0.5,

    # TVF args
    'use_tvf': True,
    'tvf_value_distribution': 'uniform',    # could improve this to advanced later on...
    'tvf_horizon_distribution': 'uniform',  # could improve this to advanced later on...
    'tvf_horizon_scale': 'wide',
    'tvf_time_scale': 'default',
    'tvf_hidden_units': 256,
    'tvf_value_samples': 32,
    'tvf_horizon_samples': 32,
    'tvf_mode': 'adaptive',                 # << not expecting this... ?
    'tvf_n_step': 20,
    'tvf_coef': 2.0,
    'tvf_soft_anchor': 1.0,
    'tvf_max_horizon': 9000,

    'gamma': 0.999,
    'tvf_gamma': 0.999,

 }


# early guesses from hps
v15_early_lh_args = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'epochs': 50,
    'export_video': False,                  # in general not needed

    # PPO args
    'max_grad_norm': 32.0,
    'agents': 512,
    'n_steps': 256,
    'policy_mini_batch_size': 512,
    'value_mini_batch_size': 256,
    'value_epochs': 2,
    'policy_epochs': 3,
    'distill_epochs': 1,
    'target_kl': 0.02,
    'ppo_epsilon': 0.125,
    'value_lr': 2.5e-4,
    'policy_lr': 1.0e-4,
    'entropy_bonus': 0.02,
    'time_aware': True,
    'distill_beta': 0.5,

    # TVF args
    'use_tvf': True,
    'tvf_value_distribution': 'advanced',   # could improve this to advanced later on...
    'tvf_horizon_distribution': 'advanced',  # could improve this to advanced later on...
    'tvf_horizon_scale': 'wide',
    'tvf_time_scale': 'default',
    'tvf_hidden_units': 1024,               # more are needed
    'tvf_value_samples': 50,
    'tvf_horizon_samples': 32,              # can still get away with very few (it seems)
    'tvf_mode': 'adaptive',                 # << not expecting this... ?
    'tvf_n_step': 20,
    'tvf_coef': 2.0,
    'tvf_soft_anchor': 0.2,
    'tvf_max_horizon': 50000,

    'gamma': 0.99997,
    'tvf_gamma': 0.99997,

 }

# guess from a good run at 30m
# this is heavy weight... might try to figure out a faster version for experimentation
v15_lh_args_v2 = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'epochs': 50,
    'export_video': False,                  # in general not needed

    # PPO args
    'max_grad_norm': 50.0,
    'agents': 256,
    'n_steps': 768,
    'policy_mini_batch_size': 256,
    'value_mini_batch_size': 512,
    'value_epochs': 3,
    'policy_epochs': 3,
    'distill_epochs': 2,
    'target_kl': 0.07,
    'ppo_epsilon': 0.1,
    'value_lr': 2.5e-4,
    'policy_lr': 1.0e-4,
    'entropy_bonus': 0.015,
    'time_aware': True,
    'distill_beta': 0.35,

    # TVF args
    'use_tvf': True,
    'tvf_value_distribution': 'uniform',
    'tvf_horizon_distribution': 'uniform',
    'tvf_horizon_scale': 'wider',
    'tvf_time_scale': 'wider',
    'tvf_hidden_units': 256,               # more are needed
    'tvf_value_samples': 128,
    'tvf_horizon_samples': 64,
    'tvf_mode': 'adaptive',                 # << not expecting this... ?
    'tvf_n_step': 40,
    'tvf_coef': 2.0,
    'tvf_soft_anchor': 2.0,
    'tvf_max_horizon': 10000,

    'gamma': 0.99997,
    'tvf_gamma': 0.99997,

 }

# tweaked mostly to be faster...
# also reduced entropy_coef to match fixed code (where entropy is ~5x smaller)
# these settings were essentially broken due to kl targeting being too low.
v15_lh_args_v3 = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'epochs': 50,
    'export_video': False,

    # PPO args
    'max_grad_norm': 10.0,
    'agents': 512,                          # maybe reduce this to 256 due to memory contraints...
    'n_steps': 512,
    'policy_mini_batch_size': 256,
    'value_mini_batch_size': 256,
    'value_epochs': 2,
    'policy_epochs': 3,
    'distill_epochs': 1,
    'target_kl': 0.01,                      # this is way too low...
    'ppo_epsilon': 0.15,
    'value_lr': 2.5e-4,
    'policy_lr': 1.0e-4,
    'entropy_bonus': 0.002,                 # seems too low?
    'time_aware': True,
    'distill_beta': 2.0,

    # TVF args
    'use_tvf': True,
    'tvf_value_distribution': 'advanced',
    'tvf_horizon_distribution': 'uniform',
    'tvf_horizon_scale': 'wider',
    'tvf_time_scale': 'wider',
    'tvf_hidden_units': 1024,
    'tvf_value_samples': 128,
    'tvf_horizon_samples': 64,
    'tvf_mode': 'adaptive',                # << not expecting this... ?
    'tvf_n_step': 100,                     # big guess...
    'tvf_coef': 2.0,
    'tvf_soft_anchor': 0.1,
    'tvf_max_horizon': 50000,

    'gamma': 0.99997,
    'tvf_gamma': 0.99997,

 }

# these are really just an 'informed' guess
v16_lh_args = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'epochs': 50,
    'export_video': False,

    # PPO args
    'max_grad_norm': 25.0,
    'agents': 256,                          # want this to be higher, but memory constrained...
    'n_steps': 256,                         # large n_steps might be needed for long horizon?
    'policy_mini_batch_size': 512,
    'value_mini_batch_size': 256,
    'policy_epochs': 3,
    'value_epochs': 2,
    'distill_epochs': 1,
    'target_kl': 0.1,
    'ppo_epsilon': 0.2,
    'value_lr': 2.5e-4,
    'policy_lr': 1.0e-4,
    'entropy_bonus': 0.01,
    'time_aware': True,
    'distill_beta': 1.0,

    # TVF args
    'use_tvf': True,
    'tvf_value_distribution': 'advanced',
    'tvf_horizon_distribution': 'advanced',
    'tvf_horizon_scale': 'wider',
    'tvf_time_scale': 'wider',
    'tvf_hidden_units': 256,               # more is probably better, but we need just ok for the moment
    'tvf_value_samples': 128,              # again, more is probably better...
    'tvf_horizon_samples': 128,
    'tvf_mode': 'adaptive',                # adaptive seems like a good tradeoff
    'tvf_n_step': 40,                      # perhaps experiment with higher later on?
    'tvf_coef': 2.0,                       # this is an important parameter...
    'tvf_soft_anchor': 0.1,                # isn't this essentially off?
    'tvf_max_horizon': 30000,              # 50k is actually better, probably because it emphasises the final value more

    'gamma': 0.99997,
    'tvf_gamma': 0.99997,

 }


def add_job(experiment_name, run_name, priority=0, chunk_size:int=10, default_params=None, score_threshold=None, **kwargs):

    if default_params is not None:
        for k,v in default_params.items():
            if k not in kwargs:
                kwargs[k]=v

    if "device" not in kwargs:
        kwargs["device"] = DEVICE

    job = Job(experiment_name, run_name, priority, chunk_size, kwargs)

    if score_threshold is not None and chunk_size > 0:
        job_details = job.get_details()
        if job_details is not None and 'score' in job_details:
            modified_kwargs = kwargs.copy()
            chunks_completed = job_details['completed_epochs'] / chunk_size
            if job_details['score'] < score_threshold * chunks_completed and chunks_completed > 0.75:
                modified_kwargs["epochs"] = chunk_size
            job = Job(experiment_name, run_name, priority, chunk_size, modified_kwargs)

    job_list.append(job)
    return job

def get_run_folders(experiment_name, run_name):
    """ Returns the paths for given experiment and run, or empty list if not found. """

    path = os.path.join(OUTPUT_FOLDER, experiment_name)
    if not os.path.exists(path):
        return []

    result = []

    for file in os.listdir(path):
        name = os.path.split(file)[-1]
        this_run_name = name[:-(8+3)]  # crop off the id code.
        if this_run_name == run_name:
            result.append(os.path.join(path, name))
    return result

def copy_source_files(source, destination, force=False):
    """ Copies all source files from source path to destination. Returns path to destination training script. """
    try:

        destination_train_script = os.path.join(destination, "train.py")

        if not force and os.path.exists(destination_train_script):
            return destination_train_script
        # we need to copy across train.py and then all the files under rl...
        os.makedirs(os.path.join(destination, "rl"), exist_ok=True)
        if platform.system() == "Windows":
            copy_command = "copy"
        else:
            copy_command = "cp"

        os.system("{} {} '{}'".format(copy_command, os.path.join(source, "train.py"), os.path.join(destination, "train.py")))
        os.system("{} {} '{}'".format(copy_command, os.path.join(source, "rl", "*.py"), os.path.join(destination, "rl")))

        return destination_train_script
    except Exception as e:
        print("Failed to copy training file to log folder.", e)
        return None


class Job:

    """
    Note: we don't cache any of the properties here as other worker may modify the filesystem, so we need to always
    use the up-to-date version.
    """

    # class variable to keep track of insertion order.
    id = 0

    def __init__(self, experiment_name, run_name, priority, chunk_size:int, params):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.priority = priority
        self.params = params
        self.id = Job.id
        self.chunk_size = chunk_size
        Job.id += 1

    def __lt__(self, other):
         return self._sort_key < other._sort_key

    @property
    def _sort_key(self):

        status = self.get_status()

        priority = self.priority

        # make running tasks appear at top...
        if status == "running":
            priority += 1000

        # if "search" in self.experiment_name.lower():
        #     # with search we want to make sure we complete partial runs first
        #     priority = priority + self.get_completed_epochs()
        # else:
        #     priority = priority - self.get_completed_epochs()

        priority = priority - self.get_completed_epochs()

        return (-priority, self.get_completed_epochs(), self.experiment_name, self.id)

    def get_path(self):
        # returns path to this job. or none if not found
        paths = self.get_paths()
        return paths[0] if len(paths) > 0 else None

    def get_paths(self):
        # returns list of paths for this jo
        return get_run_folders(self.experiment_name, self.run_name)

    def get_status(self):
        """
        Returns job status

        "": Job has not been started
        "clash": Job has multiple folders matching job name
        "running" Job is currently running
        "pending" Job has been started not not currently active
        "stale: Job has a lock that has not been updated in 30min

        """

        paths = self.get_paths()
        if len(paths) >= 2:
            return "clash"

        if len(paths) == 0:
            return ""

        status = ""

        path = paths[0]

        if os.path.exists(os.path.join(path, "params.txt")):
            status = "pending"

        if os.path.exists(os.path.join(path, "lock.txt")):
            status = "running"

        details = self.get_details()
        if details is not None and details["fraction_complete"] >= 1.0:
            status = "done"

        if status in ["running"] and self.minutes_since_modified() > 30:
            status = "stale"

        return status

    def minutes_since_modified(self):
        path = self.get_path()
        if path is None:
            return -1
        if os.path.exists(os.path.join(path, "lock.txt")):
            last_modifed = os.path.getmtime(os.path.join(path, "lock.txt"))
            return (time.time()-last_modifed)/60
        return -1

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
                details["max_epochs"] = self.params["epochs"]
                details["eta"] = (details["max_epochs"] - details["completed_epochs"]) * 1e6 / details["fps"]

            return details
        except:
            return None

    def get_completed_epochs(self):
        details = self.get_details()
        if details is not None:
            return details["completed_epochs"]
        else:
            return 0

    def run(self, chunk_size:int):

        self.params["output_folder"] = OUTPUT_FOLDER

        experiment_folder = os.path.join(OUTPUT_FOLDER, self.experiment_name)

        # make the destination folder...
        if not os.path.exists(experiment_folder):
            print("Making new experiment folder {experiment_folder}")
            os.makedirs(experiment_folder, exist_ok=True)

        # copy script across if needed.
        train_script_path = copy_source_files("./", experiment_folder)

        self.params["experiment_name"] = self.experiment_name
        self.params["run_name"] = self.run_name

        details = self.get_details()

        if details is not None and details["completed_epochs"] > 0:
            # restore if some work has already been done.
            self.params["restore"] = True
            print(f"Found restore point {self.get_path()} at epoch {details['completed_epochs']}")
        else:
            print(f"No restore point found for path {self.get_path()}")

        if chunk_size > 0:
            # work out the next block to do
            if details is None:
                next_chunk = chunk_size
            else:
                next_chunk = (round(details["completed_epochs"] / chunk_size) * chunk_size) + chunk_size
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

        if status in ["", "pending"]:

            job.get_params()

            job.run(chunk_size=job.chunk_size)
            return

def comma(x):
    if type(x) is int or (type(x) is float and x >= 100):
        postfix = ''
        # if x > 100*1e6:
        #     postfix = 'M'
        #     x /= 1e6
        # elif x > 100*1e3:
        #     postfix = 'K'
        #     x /= 1e3
        return f"{int(x):,}{postfix}"
    elif type(x) is float:
        return f"{x:.1f}"
    else:
        return str(x)

def show_experiments(filter_jobs=None, all=False):
    job_list.sort()
    print("-" * 169)
    print("{:^10}{:<20}{:<60}{:>10}{:>10}{:>10}{:>10}{:>10} {:<15} {:>6}".format("priority", "experiment_name", "run_name", "complete", "status", "eta", "fps", "score", "host", "ping"))
    print("-" * 169)
    for job in job_list:

        if filter_jobs is not None and not filter_jobs(job):
                continue

        status = job.get_status()

        if status == "done" and not all:
            continue

        details = job.get_details()

        if details is not None:
            percent_complete = "{:.1f}%".format(details["fraction_complete"]*100)
            eta_hours = "{:.1f}h".format(details["eta"] / 60 / 60)
            score = details["score"]
            if score is None: score = 0
            score = comma(score)

            if status == "running":
                host = details["host"][:8] + "/" + details.get("device", "?")
            else:
                host = ""

            fps = int(details["fps"])
            minutes = job.minutes_since_modified()
            ping = f"{minutes:.0f}" if minutes >= 0 else ""
        else:
            percent_complete = ""
            eta_hours = ""
            score = ""
            host = ""
            fps = ""
            ping = ""

        print("{:^10}{:<20}{:<60}{:>10}{:>10}{:>10}{:>10}{:>10} {:<15} {:>6}".format(
            job.priority, job.experiment_name[:19], job.run_name, percent_complete, status, eta_hours, comma(fps), comma(score), host, ping))


def show_fps(filter_jobs=None):
    job_list.sort()

    fps = {}

    for job in job_list:

        if filter_jobs is not None and not filter_jobs(job):
            continue

        status = job.get_status()

        if status == "running":
            details = job.get_details()
            if details is None:
                continue
            host = details["host"]
            if host not in fps:
                fps[host] = 0
            fps[host] += int(details["fps"])

    for k,v in fps.items():
        print(f"{k:<20} {v:,.0f} FPS")


def random_search(
        run:str,
        main_params: dict,
        search_params:dict,
        envs: list,
        score_thresholds: list,
        count: int = 128,
        process_up_to=None,
        base_seed=0,

):
    """
    Improved random search:
    for consistantancy random seed is now based on key.
    values are evenly distributed over range then shuffled
    """

    assert len(envs) == len(score_thresholds)

    # note: for categorical we could do better creating a list with the correct proportions then shuffeling it
    # the last run had just 4 wide out of 32 when 10 or 11 were expected...
    # one disadvantage of this is that we can not change the count after the search has started. (although we could run it twice I guess?)

    import numpy as np
    import hashlib

    def smart_round_sig(x, sig=4):
        if int(x) == x:
            return x
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

    even_dist_samples = {}

    # this method makes sure categorical samples are well balanced
    for k, v in search_params.items():
        seed = hashlib.sha256(k.encode("UTF-8")).digest()
        random.seed(int.from_bytes(seed, "big")+base_seed)
        if type(v) is Categorical:
            samples = []
            for _ in range(math.ceil(count/len(v._values))):
                samples.extend(v._values)
        elif type(v) is Uniform:
            samples = np.linspace(v._min, v._max, count)
        elif type(v) is LogUniform:
            samples = np.logspace(v._min, v._max, base=math.e, num=count)
        else:
            raise TypeError()

        random.shuffle(samples)
        even_dist_samples[k] = samples[:count]

    for i in range(process_up_to or count):
        params = {}
        for k, v in search_params.items():
            params[k] = even_dist_samples[k][i]
            if type(v) in [Uniform, LogUniform] and v._force_int:
                params[k] = int(params[k])
            if type(params[k]) in [float, np.float64]:
                params[k] = smart_round_sig(params[k])

        # agents must divide workers (which we assume divides 8)
        params['agents'] = (params['agents'] // 8) * 8

        # make sure mini_batch_size is not larger than batch_size
        params["policy_mini_batch_size"] = min(params["agents"] * params["n_steps"], params["policy_mini_batch_size"])
        params["value_mini_batch_size"] = min(params["agents"] * params["n_steps"], params["value_mini_batch_size"])

        for env_name, score_threshold in zip(envs, score_thresholds):
            main_params['env_name'] = env_name
            add_job(run, run_name=f"{i:04d}_{env_name}", chunk_size=10, score_threshold=score_threshold, **main_params, **params)


def nice_format(x):
    if type(x) is str:
        return f'"{x}"'
    if x is None:
        return "None"
    if type(x) in [int, float, bool]:
        return str(x)

    return f'"{x}"'

# ---------------------------------------------------------------------------------------------------------

def setup_experiments_13_eval():

    for env_name in ["Alien", "BankHeist", "CrazyClimber"]:
        pass

class Categorical():

    def __init__(self, *args):
        self._values = args

    def sample(self):
        return random.choice(self._values)


class Uniform():
    def __init__(self, min, max, force_int=False):
        self._min = min
        self._max = max
        self._force_int = force_int

    def sample(self):
        r = (self._max-self._min)
        a = self._min
        result = a + random.random() * r
        return int(result) if self._force_int else result

class LogUniform():
    def __init__(self, min, max, force_int=False):
        self._min = math.log(min)
        self._max = math.log(max)
        self._force_int = force_int

    def sample(self):
        r = (self._max - self._min)
        a = self._min
        result = math.exp(a + random.random() * r)
        return int(result) if self._force_int else result

def random_search_15_tvf():

    # this is the long horizon HPS
    # let's go for broke... (30k effective horizion)
    # changes:
    # * removed first and last
    # * fixed advanced exponential mode
    # * fixed policy epochs limited to 1.0
    # * tighten most hps


    main_params = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'export_video': False, # save some space...
        'use_tvf': True,
        'gamma': 0.99997,
        'tvf_gamma': 0.99997,
        'epochs': 50,
        'use_compression': True,
        'priority': 400,
    }

    search_params = {

        # ppo params
        'max_grad_norm':    LogUniform(10, 100),
        'agents':           LogUniform(256, 2048, force_int=True),
        'n_steps':          LogUniform(128, 1024, force_int=True),

        'policy_mini_batch_size': Categorical(256, 512, 1024),
        'value_mini_batch_size': Categorical(256, 512, 1024),

        'value_epochs':     Categorical(2, 3),
        'policy_epochs':    Categorical(3),
        'distill_epochs':   Categorical(1, 2),
        'distill_beta':     LogUniform(0.3, 3),     # << need to know
        'target_kl':        LogUniform(0.001, 1.0), # << need to know
        'ppo_epsilon':      Uniform(0.03, 0.3),     # << need to know
        'value_lr':         Categorical(1e-4, 2.5e-4),
        'policy_lr':        Categorical(1e-4, 2.5e-4),
        'entropy_bonus':    LogUniform(0.01, 0.04),

        # tvf params
        'tvf_coef':         LogUniform(1, 3),       # this might change with higher horizon
        'tvf_mode':         Categorical("exponential", "adaptive", "nstep"),
        'tvf_n_step':       LogUniform(20, 80, force_int=True),

        'time_aware':       Categorical(True, False),
        'tvf_max_horizon':  LogUniform(10000, 90000, force_int=True), # there's a bit of an advantage to having horizon much longer than tl.
        'tvf_value_samples': LogUniform(16, 256, force_int=True),
        'tvf_horizon_samples': LogUniform(16, 256, force_int=True),
        'tvf_value_distribution': Categorical("uniform", "advanced"),
        'tvf_horizon_distribution': Categorical("uniform", "advanced"),
        'tvf_hidden_units': LogUniform(32, 1024, force_int=True), # << big unknown...
        'tvf_soft_anchor': LogUniform(0.1, 3),      # << need to know (now that first and last is removed
        'tvf_horizon_scale': Categorical("wide", "wider"),
        'tvf_time_scale': Categorical("wide", "wider", "zero"),
    }

    # score threshold should be 200, but I want some early good results...
    random_search(
        "TVF_15_Search_30k",
        main_params,
        search_params,
        count=32,
        process_up_to=16,
        envs=['BattleZone', 'DemonAttack', 'Amidar'],
        # set roughly to 1x random
        score_thresholds=[2000*2, 150*2, 5*2],
    )


def random_search_16_tvf():

    # a few tweaks
    #  * better system for restoring from checkpoint (much faster when there are lots of environments)


    main_params = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'export_video': False, # save some space...
        'use_tvf': True,
        'gamma': 0.99997,
        'tvf_gamma': 0.99997,
        'epochs': 50,
        'use_compression': True,
        'priority': 0,
    }

    search_params = {

        # ppo params
        'max_grad_norm':    LogUniform(10, 100),
        'agents':           LogUniform(256, 2048, force_int=True),
        'n_steps':          LogUniform(128, 1024, force_int=True),

        'policy_mini_batch_size': Categorical(256, 512, 1024),
        'value_mini_batch_size': Categorical(256, 512, 1024),

        'value_epochs':     Categorical(2, 3),
        'policy_epochs':    Categorical(3),
        'distill_epochs':   Categorical(1, 2),
        'distill_beta':     LogUniform(0.3, 3),     # << need to know
        'target_kl':        LogUniform(0.001, 1.0), # << need to know
        'ppo_epsilon':      Uniform(0.03, 0.3),     # << need to know
        'value_lr':         Categorical(1e-4, 2.5e-4),
        'policy_lr':        Categorical(1e-4, 2.5e-4),
        'entropy_bonus':    LogUniform(0.01, 0.04),

        # tvf params
        'tvf_coef':         LogUniform(1, 3),       # this might change with higher horizon
        'tvf_mode':         Categorical("exponential", "adaptive", "nstep"),
        'tvf_n_step':       LogUniform(20, 80, force_int=True),

        'time_aware':       Categorical(True, False),
        'tvf_max_horizon':  LogUniform(10000, 90000, force_int=True), # there's a bit of an advantage to having horizon much longer than tl.
        'tvf_value_samples': LogUniform(16, 256, force_int=True),
        'tvf_horizon_samples': LogUniform(16, 256, force_int=True),
        'tvf_value_distribution': Categorical("uniform", "advanced"),
        'tvf_horizon_distribution': Categorical("uniform", "advanced"),
        'tvf_hidden_units': LogUniform(32, 1024, force_int=True), # << big unknown...
        'tvf_soft_anchor': LogUniform(0.1, 3),      # << need to know (now that first and last is removed
        'tvf_horizon_scale': Categorical("wide", "wider"),
        'tvf_time_scale': Categorical("wide", "wider", "zero"),
    }

    # score threshold should be 200, but I want some early good results...
    random_search(
        "TVF_16_Search_30k",
        main_params,
        search_params,
        count=32,
        process_up_to=16,
        envs=['Amidar', 'BattleZone', 'DemonAttack', ],
        # set roughly to 1x random
        score_thresholds=[5*2, 2000*2, 150*2],
    )


def setup_experiments_16():
    for tvf_n_step in range(20, 40, 80):
        add_job(
            f"TVF_16_Regression",
            env_name="DemonAttack",
            run_name=f"tvf_n_step={tvf_n_step}",
            default_params=v16_lh_args,
            epochs=50,
            priority=200,
        )

    if True:
        add_job(
            f"TVF_16_Distill",
            env_name="DemonAttack",
            run_name=f"tvf_full",
            default_params=v16_lh_args,
            epochs=50,
            priority=100,
        )

        add_job(
            f"TVF_16_Distill",
            env_name="DemonAttack",
            run_name=f"tvf_300",
            default_params=v16_lh_args,
            tvf_max_horizon=3000,
            gamma=0.99,
            tvf_gamma=0.99,
            epochs=50,
            priority=100,
        )

        add_job(
            f"TVF_16_Distill",
            env_name="DemonAttack",
            run_name=f"tvf_ext",
            default_params=v16_lh_args,
            tvf_force_ext_value_distill=True,
            epochs=50,
            priority=100,
        )

        add_job(
            f"TVF_16_Distill",
            env_name="DemonAttack",
            run_name=f"tvf_off",
            default_params=v16_lh_args,
            distill_epochs=0,
            epochs=50,
            priority=100,
        )

        add_job(
            f"TVF_16_Distill",
            env_name="DemonAttack",
            run_name=f"ppg_99997",
            use_tvf=False,
            default_params=v16_lh_args,
            epochs=50,
            priority=1000,
        )

        add_job(
            f"TVF_16_Distill",
            env_name="DemonAttack",
            run_name=f"ppo_99997",
            use_tvf=False,
            distill_epochs=0,
            default_params=v16_lh_args,
            epochs=50,
            priority=100,
        )

        add_job(
            f"TVF_16_Distill",
            env_name="DemonAttack",
            run_name=f"ppg_99",
            use_tvf=False,
            gamma=0.99,
            default_params=v16_lh_args,
            epochs=50,
            priority=1000,
        )

        add_job(
            f"TVF_16_Distill",
            env_name="DemonAttack",
            run_name=f"ppo_99",
            use_tvf=False,
            gamma=0.99,
            distill_epochs=0,
            default_params=v16_lh_args,
            epochs=50,
            priority=100,
        )

    # entropy annealing
    for alpha in [0, 1.0]:
        for beta in [0, -0.5]:
            theta = 1.0
            add_job(
                f"TVF_16_EBS_CrazyClimber",
                env_name="CrazyClimber",
                run_name=f"alpha={alpha} beta={beta} theta={theta}",
                default_params=v16_lh_args,
                eb_alpha=alpha,
                eb_beta=beta,
                eb_theta=theta,
                epochs=50,
                priority=200,
            )
    for theta in [0, 1, 1/2, 1/5]:
            alpha = 1
            beta = -0.5
            add_job(
                f"TVF_16_EBS_CrazyClimber",
                env_name="CrazyClimber",
                run_name=f"alpha={alpha} beta={beta} theta={theta}",
                default_params=v16_lh_args,
                eb_alpha=alpha,
                eb_beta=beta,
                eb_theta=theta,
                epochs=50,
                priority=200,
            )



    # try on skiing
    add_job(
        f"TVF_16_EBS_Skiing",
        env_name="Skiing",
        run_name=f"alpha=1.0 beta=-0.5 theta=1",
        default_params=v16_lh_args,
        eb_alpha=1.0,
        eb_beta=-0.5,
        eb_theta=1.0,
        epochs=200,
        priority=200,
    )


def setup_experiments_15():
    add_job(
        f"TVF_15_Regression",
        env_name="DemonAttack",
        run_name=f"run=1",
        default_params=v15_early_lh_args,
        epochs=200,
        priority=50,
    )

    # random games
    #for env_name in ['Breakout', 'Seaquest', 'Skiing', 'MontezumaRevenge']:
    for env_name in ['Breakout', 'Seaquest', 'Skiing']:
        add_job(
            f"TVF_15_{env_name}",
            env_name=env_name,
            run_name=f"tfv_50k",
            default_params=v15_early_lh_args,
            epochs=200 if env_name in ["Breakout", "Skiing"] else 50,
            priority=200,
            chunk_size=50,
        )
        add_job(
            f"TVF_15_{env_name}",
            env_name=env_name,
            run_name=f"tvf_3k",
            default_params=v15_args,
            epochs=50,
            priority=50,
        )

        # distill not supported yet on ppg...

        # add_job(
        #     f"TVF_15_{env_name}",
        #     env_name=env_name,
        #     run_name=f"ppg_999",
        #     use_tvf=False,
        #     gamma=0.999,
        #     default_params=v15_args,
        #     epochs=50,
        #     priority=20,
        # )
        # add_job(
        #     f"TVF_15_{env_name}",
        #     env_name=env_name,
        #     run_name=f"ppg_99997",
        #     use_tvf=False,
        #     gamma=0.99997,
        #     default_params=v15_args,
        #     epochs=50,
        #     priority=20,
        # )

    # this is just making sure that saving and loading works ok.
    if True:
        add_job(
            f"TVF_15_Restore",
            env_name="DemonAttack",
            run_name=f"tvf_3k",
            default_params=v15_args,
            agents=32,
            epochs=10,
            priority=200,
            chunk_size=1,
        )

    # value heads
    for tvf_n_value_heads in [1, 4, 16]:
        add_job(
            f"TVF_15_ValueHeads",
            env_name="DemonAttack",
            run_name=f"tvf_n_value_heads={tvf_n_value_heads}",
            default_params=v15_lh_args_v2,
            tvf_n_value_heads=tvf_n_value_heads,
            epochs=10,
            priority=200,
        )

    # entropy annealing
    # for alpha in [0, 1.0]:
    #     for beta in [0, -0.5]:
    #         add_job(
    #             f"TVF_15_EB_DemonAttack",
    #             env_name="DemonAttack",
    #             run_name=f"alpha={alpha} beta={beta} theta=1",
    #             default_params=v15_lh_args_v3,
    #             eb_alpha=alpha,
    #             eb_beta=beta,
    #             eb_theta=1.0,
    #             epochs=50,
    #             priority=200,
    #         )
    # for theta in [0, 1, 1/2, 1/5]:
            # alpha = 1
            # beta = -0.5
    #         add_job(
    #             f"TVF_15_EB_DemonAttack",
    #             env_name="DemonAttack",
    #             run_name=f"alpha={alpha} beta={beta} theta={theta}",
    #             default_params=v15_lh_args_v3,
    #             eb_alpha=alpha,
    #             eb_beta=beta,
    #             eb_theta=theta,
    #             epochs=50,
    #             priority=200,
    #         )

    # had issues
    if False:
        # quick check on new distill code...
        # add PPO soon...
        add_job(
            f"TVF_15_Distill",
            env_name="DemonAttack",
            run_name=f"tvf_full",
            default_params=v15_lh_args_v3,
            epochs=50,
            priority=100,
        )

        add_job(
            f"TVF_15_Distill",
            env_name="DemonAttack",
            run_name=f"tvf_ext",
            default_params=v15_lh_args_v3,
            tvf_force_ext_value_distill=True,
            epochs=50,
            priority=100,
        )

        add_job(
            f"TVF_15_Distill",
            env_name="DemonAttack",
            run_name=f"tvf_off",
            default_params=v15_lh_args_v3,
            distill_epochs=0,
            epochs=50,
            priority=100,
        )

        add_job(
            f"TVF_15_Distill",
            env_name="DemonAttack",
            run_name=f"ppo_distill",
            use_tvf=False,
            default_params=v15_lh_args_v3,
            epochs=50,
            priority=100,
        )

        add_job(
            f"TVF_15_Distill",
            env_name="DemonAttack",
            run_name=f"ppo",
            use_tvf=False,
            distill_epochs=0,
            default_params=v15_lh_args_v3,
            epochs=50,
            priority=100,
        )

    # try on skiing
    add_job(
        f"TVF_15_EB_Skiing",
        env_name="Skiing",
        run_name=f"alpha=1.0 beta=-0.5 theta=1",
        default_params=v15_lh_args_v3,
        eb_alpha=1.0,
        eb_beta=-0.5,
        eb_theta=1.0,
        epochs=200,
        priority=200,
    )

    # value heads
    add_job(
        f"TVF_15_Breakout_MVH_EXP",
        env_name="Breakout",
        run_name=f"reference",
        tvf_horizon_scale='centered',  # see if this is enough...
        tvf_time_scale='centered',
        tvf_mode="exponential",
        default_params=v15_lh_args_v2,
        tvf_n_value_heads=1,
        epochs=20,
        priority=20,
    )
    for tvf_n_value_heads in [2, 4, 8]:
        add_job(
            f"TVF_15_Breakout_MVH_EXP",
            env_name="Breakout",
            run_name=f"heads={tvf_n_value_heads} (piecewise)",
            tvf_horizon_scale='centered', # see if this is enough...
            tvf_time_scale='centered',
            tvf_mode="exponential",
            default_params=v15_lh_args_v2,
            tvf_n_value_heads=tvf_n_value_heads,
            epochs=20,
            priority=20,
        )

        add_job(
            f"TVF_15_Breakout_MVH_EXP",
            env_name="Breakout",
            run_name=f"heads={tvf_n_value_heads} (step)",
            tvf_horizon_scale='zero',
            tvf_time_scale='centered',
            tvf_mode="exponential",
            default_params=v15_lh_args_v2,
            tvf_n_value_heads=tvf_n_value_heads,
            epochs=20,
            priority=20,
        )

        add_job(
            f"TVF_15_Breakout_MVH_EXP",
            env_name="Breakout",
            run_name=f"heads={tvf_n_value_heads} (linear)",
            tvf_horizon_scale='zero',
            tvf_time_scale='centered',
            tvf_mode="exponential",
            default_params=v15_lh_args_v2,
            tvf_n_value_heads=-tvf_n_value_heads,
            epochs=20,
            priority=20,
        )

        add_job(
            f"TVF_15_Breakout_MVH_EXP",
            env_name="Breakout",
            run_name=f"heads={tvf_n_value_heads} (blended)",
            tvf_horizon_scale='centered',
            tvf_time_scale='centered',
            tvf_mode="exponential",
            default_params=v15_lh_args_v2,
            tvf_n_value_heads=-tvf_n_value_heads,
            epochs=20,
            priority=20,
        )

    # breakout horizon quality (with new settings)
    for horizon, gamma in zip([300, 1000, 3000, 30000], [0.997, 0.999, 0.9997, 0.99997]):

        # try to do better on breakout...
        add_job(
            f"TVF_15_Breakout_HQ",
            env_name="Breakout",
            run_name=f"max_horizon={horizon} gamma={gamma} (v2)",
            default_params=v15_lh_args_v2,
            tvf_max_horizon=horizon,
            gamma=gamma,
            tvf_gamma=gamma,
            epochs=20,
            priority=100,
        )

        if horizon == 3000:
            # special cases to get this to work...
            add_job(
                f"TVF_15_Breakout_HQ",
                env_name="Breakout",
                run_name=f"max_horizon={horizon} gamma={gamma} (v2_tms=100)",
                default_params=v15_lh_args_v2,
                tvf_n_step=100,
                tvf_max_horizon=horizon,
                gamma=gamma,
                tvf_gamma=gamma,
                epochs=20,
                priority=100,
            )
            add_job(
                f"TVF_15_Breakout_HQ",
                env_name="Breakout",
                run_name=f"max_horizon={horizon} gamma={gamma} (v2_hu=512)",
                default_params=v15_lh_args_v2,
                tvf_hidden_units=512,
                tvf_max_horizon=horizon,
                gamma=gamma,
                tvf_gamma=gamma,
                epochs=20,
                priority=100,
            )
            add_job(
                f"TVF_15_Breakout_HQ",
                env_name="Breakout",
                run_name=f"max_horizon={horizon} gamma={gamma} (samples=512)",
                default_params=v15_lh_args_v2,
                tvf_value_samples=512,
                tvf_horizon_samples=512,
                tvf_max_horizon=horizon,
                gamma=gamma,
                tvf_gamma=gamma,
                epochs=20,
                priority=100,
            )
            add_job(
                f"TVF_15_Breakout_HQ",
                env_name="Breakout",
                run_name=f"max_horizon={horizon} gamma={gamma} (n_step=512)",
                default_params=v15_lh_args_v2,
                tvf_max_horizon=horizon,
                tvf_mode="nstep",
                tvf_n_step=512,
                gamma=gamma,
                tvf_gamma=gamma,
                epochs=20,
                priority=100,
            )
            add_job(
                f"TVF_15_Breakout_HQ",
                env_name="Breakout",
                run_name=f"max_horizon={horizon} gamma={gamma} (adaptive=512)",
                default_params=v15_lh_args_v2,
                tvf_max_horizon=horizon,
                tvf_mode="adaptive",
                tvf_n_step=512,
                gamma=gamma,
                tvf_gamma=gamma,
                epochs=20,
                priority=100,
            )

        add_job(
            f"TVF_15_Breakout_HQ",
            env_name="Breakout",
            run_name=f"max_horizon={horizon} gamma={gamma} (v2-centered)",
            default_params=v15_lh_args_v2,
            tvf_horizon_scale='centered',
            tvf_max_horizon=horizon,
            gamma=gamma,
            tvf_gamma=gamma,
            epochs=20,
            priority=100,
        )

        add_job(
            f"TVF_15_Breakout_HQ",
            env_name="Breakout",
            run_name=f"max_horizon={horizon} gamma={gamma} (v2-log)",
            tvf_horizon_scale='log',
            default_params=v15_lh_args_v2,
            tvf_max_horizon=horizon,
            gamma=gamma,
            tvf_gamma=gamma,
            epochs=20,
            priority=100,
        )

        add_job(
            f"TVF_15_Breakout_HQ",
            env_name="Breakout",
            run_name=f"max_horizon={horizon} gamma={gamma} (v2-exp)",
            tvf_mode="exponential",
            default_params=v15_lh_args_v2,
            tvf_max_horizon=horizon,
            gamma=gamma,
            tvf_gamma=gamma,
            epochs=20,
            priority=100,
        )

        add_job(
            f"TVF_15_Breakout_HQ",
            env_name="Breakout",
            run_name=f"max_horizon={horizon} gamma={gamma} (v2-nt)",
            tvf_time_scale="zero", # this might be confusing things...?
            default_params=v15_lh_args_v2,
            tvf_max_horizon=horizon,
            gamma=gamma,
            tvf_gamma=gamma,
            epochs=20,
            priority=100,
        )




if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    id = 0
    job_list = []
    random_search_15_tvf()
    setup_experiments_16()

    if len(sys.argv) == 1:
        experiment_name = "show"
    else:
        experiment_name = sys.argv[1]

    if experiment_name == "show_all":
        show_experiments(all=True)
    elif experiment_name == "show":
        show_experiments()
    elif experiment_name == "fps":
        show_fps()
    elif experiment_name == "auto":
        run_next_experiment()
    else:
        run_next_experiment(filter_jobs=lambda x: x.experiment_name == experiment_name)