import os
import sys
import json
import time
import random
import platform
import math
import shlex

import socket
import subprocess

"""

Notes on parameters

standard_args: These are the default args for a lot of the older experiments. Should work fine for PPO, and DNA.
    for TVF use replay args.

"""

job_list = []
id = 0

ROM_FOLDER = "./roms"
os.environ["ALE_PY_ROM_DIR"] = ROM_FOLDER # set folder for atari roms.

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
WORKERS=8

ATARI_VAL = ['Krull', 'KungFuMaster', 'Seaquest']
ATARI_3 = ['BattleZone', 'Gopher', 'TimePilot']
ATARI_2 = ['Krull', 'KungFuMaster']
MYSTIC_FIVE = ['Pong', 'CrazyClimber', 'Surround', 'Breakout', 'Skiing']


ATARI_5 = ['Centipede', 'CrazyClimber', 'Krull', 'SpaceInvaders', 'Zaxxon']  # Atari5
LONG_TERM_CREDIT = ['BeamRider', 'Pong', 'Skiing', 'Surround']  # long term credit assignment from Agent57

DIVERSE_5 = ['BattleZone', 'TimePilot', "Seaquest", "Breakout", "Freeway"] # not the real atari 5..., also not 5.
DIVERSE_10 = list(set(ATARI_5 + LONG_TERM_CREDIT + ['Breakout']))

if len(sys.argv) == 3:
    DEVICE = sys.argv[2]

canonical_57 = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "BankHeist",
    "BattleZone",
    "BeamRider",
    "Berzerk",
    "Bowling",
    "Boxing",
    "Breakout",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "Defender",
    "DemonAttack",
    "DoubleDunk",
    "Enduro",
    "FishingDerby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "IceHockey",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MontezumaRevenge",
    "MsPacman",
    "NameThisGame",
    "Phoenix",
    "Pitfall",
    "Pong",
    "PrivateEye",
    "Qbert",
    "Riverraid",
    "RoadRunner",
    "Robotank",
    "Seaquest",
    "Skiing",
    "Solaris",
    "SpaceInvaders",
    "StarGunner",
    "Surround",
    "Tennis",
    "TimePilot",
    "Tutankham",
    "UpNDown",
    "Venture",
    "VideoPinball",
    "WizardOfWor",
    "YarsRevenge",
    "Zaxxon"
]


# these are the reference settings, they should not be changed
TVF_reference_args = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'architecture': 'dual',
    'export_video': False,
    'epochs': 50,
    'use_compression': 'auto',
    'warmup_period': 1000,
    'disable_ev': False,
    'seed': 0,

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
    'distill_epochs': 1,
    'distill_beta': 1.0,
    'target_kl': -1,
    'ppo_epsilon': 0.1,
    'policy_lr': 2.5e-4,
    'value_lr': 2.5e-4,
    'distill_lr': 2.5e-4,
    'entropy_bonus': 1e-3,
    'tvf_force_ext_value_distill': True,
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
    'tvf_n_step': 80,  # makes no difference...
    'tvf_exp_gamma': 2.0,  # 2.0 would be faster, but 1.5 tested slightly better.
    'tvf_coef': 0.5,
    'tvf_soft_anchor': 0,
    'tvf_exp_mode': "transformed",

    'observation_normalization': True,  # very important for DNA

    # horizon
    'gamma': 0.99997,
    'tvf_gamma': 0.99997,
    'tvf_max_horizon': 30000,
}

# at 30k horizon
DNA_reference_args = TVF_reference_args.copy()
DNA_reference_args.update({
    'use_tvf': False,
})

# at 30k horizon
PPO_reference_args = DNA_reference_args.copy()
PPO_reference_args.update({
    'use_tvf': False,
    'architecture': 'single',
})


# these are the standard args I use for most experiments.
# maybe remove this, and make it per experiment. The issue is that I can never change these settings...
standard_args = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'architecture': 'dual',
    'export_video': False,
    'epochs': 50,  # really want to see where these go...
    'use_compression': 'auto',
    'warmup_period': 1000,
    # helps on some games to make sure they are really out of sync at the beginning of training.
    'disable_ev': False,
    'seed': 0,

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
    'tvf_n_step': 80,  # makes no difference...
    'tvf_exp_gamma': 2.0,  # 2.0 would be faster, but 1.5 tested slightly better.
    'tvf_coef': 0.5,
    'tvf_soft_anchor': 0,
    'tvf_exp_mode': "transformed",

    # horizon
    'gamma': 1.0,
    'tvf_gamma': 1.0,
    'tvf_max_horizon': 30000,
}

# these enhanced args improve performance, but make the algorithm slower.
enhanced_args = standard_args.copy()
enhanced_args.update({
    'tvf_horizon_samples': 128,
    'tvf_exp_gamma': 1.5,  # 2.0 would be faster, but 1.5 tested slightly better.
})

# these enhanced args improve performance, but make the algorithm slower.
simple_args = enhanced_args.copy()
simple_args.update({
    'tvf_force_ext_value_distill': True,
})

replay_simple_args = enhanced_args.copy()
replay_simple_args.update({
    'gamma': 0.99997,
    'tvf_gamma': 0.99997,
    'replay_mixing':False,
    'distill_epochs': 1,
    'distil_period':2,
    'replay_size': 1 * 128 * 128,
    'replay_mode': "uniform",
    'dna_dual_constraint': 0.3,
    'use_compression': False, # required for replay buffer (for the moment.)
    'use_mutex': True, # faster...
    'tvf_force_ext_value_distill': True,
})

replay_full_args = replay_simple_args.copy()
replay_full_args.update({
    'tvf_force_ext_value_distill': True,
})


# these are the new default replay buffer updated to support new renamed config settings,
replay_args = enhanced_args.copy()
replay_args.update({
    'gamma': 0.99997,
    'tvf_gamma': 0.99997,
    'replay_mixing': False,
    'distill_epochs': 1,
    'distil_period': 1,
    'replay_size': 128 * 128,
    'distil_batch_size': 128 * 128 // 2,
    'replay_mode': "uniform",
    'dna_dual_constraint': 0.3,
    'use_compression': True,
    'use_mutex': True, # faster...
    'tvf_force_ext_value_distill': False,
})


def add_job(
        experiment_name,
        run_name,
        priority=0,
        chunk_size: int = 10,
        default_params=None,
        score_threshold=None,
        hostname: str = '',
        **kwargs
):

    if default_params is not None:
        for k, v in default_params.items():
            if k == "chunk_size":
                chunk_size = v
            elif k == "priority":
                priority = v
            elif k == "hostname":
                hostname = v
            else:
                if k not in kwargs:
                    kwargs[k] = v

    if "device" not in kwargs:
        kwargs["device"] = DEVICE

    if "mutex_key" not in kwargs:
        kwargs["mutex_key"] = "DEVICE"

    if kwargs.get("epochs", -1) == 0:
        # ignore runs with 0 epochs, but only if epochs is not specified.
        return

    job = Job(experiment_name, run_name, params=kwargs, priority=priority, chunk_size=chunk_size, hostname=hostname)

    if score_threshold is not None and chunk_size > 0:
        job_details = job.get_details()
        if job_details is not None and 'score' in job_details:
            modified_kwargs = kwargs.copy()
            chunks_completed = job_details['completed_epochs'] / chunk_size
            if job_details['score'] < score_threshold * chunks_completed and chunks_completed > 0.75:
                modified_kwargs["epochs"] = chunk_size
            job = Job(experiment_name, run_name, params=modified_kwargs, priority=priority, chunk_size=chunk_size, hostname=hostname)

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
        os.makedirs(os.path.join(destination, "roms"), exist_ok=True)
        if platform.system() == "Windows":
            copy_command = "copy"
        else:
            copy_command = "cp"

        os.system("{} {} '{}'".format(copy_command, os.path.join(source, "train.py"), os.path.join(destination, "train.py")))
        os.system("{} {} '{}'".format(copy_command, os.path.join(source, "rl", "*.py"), os.path.join(destination, "rl")))
        os.system("{} {} '{}'".format(copy_command, os.path.join(source, "roms", "*.bin"), os.path.join(destination, "roms")))

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

    def __init__(self, experiment_name, run_name, params, priority=0, chunk_size: int = 10, hostname=None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.priority = priority
        self.params = params
        self.id = Job.id
        self.chunk_size = chunk_size
        self.hostname = hostname
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
        # returns list of paths for this job
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
        # stub: should be 1.0, but due to a bug in one version we consider 99.9 complete.
        if details is not None and details["fraction_complete"] >= 0.999:
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
        except Exception as e:
            return None

    def get_completed_epochs(self):
        details = self.get_details()
        if details is not None:
            return details["completed_epochs"]
        else:
            return 0

    def run(self,
            chunk_size: int = 10,
            run_async=False,
            force_no_restore: bool = False,
            verbose=True,
            preamble: str = "",
            ):
        """
        Executes this job
        run_async: if true command is run in background.
        numa_ia: forces a numa group, must have numactl installed.
        """

        self.params["output_folder"] = OUTPUT_FOLDER

        experiment_folder = os.path.join(OUTPUT_FOLDER, self.experiment_name)

        # make the destination folder...
        if not os.path.exists(experiment_folder):
            if verbose:
                print(f"Making new experiment folder {experiment_folder}")
            os.makedirs(experiment_folder, exist_ok=True)

        # copy script across if needed.
        train_script_path = copy_source_files("./", experiment_folder)

        self.params["experiment_name"] = self.experiment_name
        self.params["run_name"] = self.run_name

        details = self.get_details()

        if details is not None and details["completed_epochs"] > 0 and not force_no_restore:
            # restore if some work has already been done.
            self.params["restore"] = True
            if verbose:
                print(f"Found restore point {self.get_path()} at epoch {details['completed_epochs']}")
        else:
            pass

        if chunk_size > 0:
            # work out the next block to do
            if details is None:
                next_chunk = chunk_size
            else:
                next_chunk = (round(details["completed_epochs"] / chunk_size) * chunk_size) + chunk_size
            self.params["limit_epochs"] = int(next_chunk)

        nice_params = [
            f"--{k}={nice_format(v)}" for k, v in self.params.items() if k not in ["env_name"] and v is not None
        ]

        if run_async:
            process_params = []
            for k, v in self.params.items():
                if k == "env_name":
                    continue
                process_params.append("--"+str(k))
                process_params.append(str(v))
            process = subprocess.Popen(
                [*shlex.split(preamble), "python", train_script_path, self.params["env_name"], *process_params],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return process
        else:
            python_part = preamble+" "+"python \"{}\" {}".format(train_script_path, self.params["env_name"])
            params_part = " ".join(nice_params)
            params_part_lined = "\n".join(nice_params)
            if verbose:
                print()
                print("=" * 120)
                print(bcolors.OKGREEN + self.experiment_name + " " + self.run_name + bcolors.ENDC)
                print("Running " + python_part + "\n" + params_part_lined)
                print("=" * 120)
                print()

            return_code = os.system(python_part + " " + params_part)
            if return_code != 0:
                raise Exception("Error {}.".format(return_code))


def nice_format(x):
    if type(x) is str:
        return f'"{x}"'
    if x is None:
        return "None"
    if type(x) in [int, float, bool]:
        return str(x)

    return f'"{x}"'



# ---------------------------------------------------------------------------------------------------------

class Categorical():

    def __init__(self, *args):
        self._values = args

    def sample(self):
        return random.choice(self._values)

class LinkedCategorical():

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


def run_next_experiment(filter_jobs=None):

    job_list.sort()

    for job in job_list:

        if filter_jobs is not None and not filter_jobs(job):
            continue

        if job.hostname is not None and not job.hostname in HOST_NAME:
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

    epochs, hours, ips = get_eta_stats()
    if hours < 0.75:
        nice_eta_time = f"{hours*60:.0f} minutes"
    elif round(hours) == 1:
        nice_eta_time = f"{hours:.0f} hour"
    elif hours < 24:
        nice_eta_time = f"{hours:.0f} hours"
    else:
        nice_eta_time = f"{hours/24:.1f} days"

    print(f"Epochs: {round(epochs):,} IPS:{round(ips):,} ETA:{nice_eta_time}")

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
            job.priority, job.experiment_name[:19], job.run_name[:60], percent_complete, status, eta_hours, comma(fps),
            comma(score), host, ping))


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

def get_eta_stats():

    total_epochs_to_go = 0

    total_ips = 0

    for job in job_list:
        details = job.get_details()
        if details is None:
            total_epochs_to_go += job.params["epochs"]
        else:
            total_epochs_to_go += max(0, job.params["epochs"] - details["completed_epochs"])
        if job.get_status() == "running" and details is not None:
            total_ips += details["fps"]

    if total_ips > 0:
        hours_remaining = total_epochs_to_go * 1e6 / total_ips / (60*60)
    else:
        hours_remaining = 0

    return total_epochs_to_go, hours_remaining, total_ips


def random_search(
        run:str,
        main_params: dict,
        search_params:dict,
        envs: list,
        score_thresholds: list=None,
        count: int = 64,
        process_up_to=None,
        base_seed=0,
        hook=None,
        priority=0,
):
    """
    Improved random search:
    for consistantancy random seed is now based on key.
    values are evenly distributed over range then shuffled
    """

    assert score_thresholds is None or (len(envs) == len(score_thresholds))

    # note: for categorical we could do better creating a list with the correct proportions then shuffeling it
    # the last run had just 4 wide out of 32 when 10 or 11 were expected...
    # one disadvantage of this is that we can not change the count after the search has started. (although we could run it twice I guess?)

    import numpy as np
    import hashlib

    def smart_round_sig(x, sig=8):
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
            raise TypeError(f"Type {type(v)} is invalid.")

        random.shuffle(samples)
        even_dist_samples[k] = samples[:count]

    for i in range(process_up_to or count):
        sample_params = {}
        for k, v in search_params.items():
            sample_params[k] = even_dist_samples[k][i]
            if type(v) in [Uniform, LogUniform] and v._force_int:
                sample_params[k] = int(sample_params[k])
            if type(sample_params[k]) in [float, np.float64]:
                sample_params[k] = smart_round_sig(sample_params[k])

        def get_setting(x):
            if x in sample_params:
                return sample_params[x]
            else:
                return main_params[x]

        # agents must divide workers
        if "agents" in sample_params:
            sample_params["agents"] = (sample_params["agents"] // WORKERS) * WORKERS

        # convert negative learning rates to annealing
        for key in ["policy_lr", "value_lr", "distill_lr"]:
            if key in sample_params and sample_params[key] < 0:
                sample_params[key] = -sample_params[key]
                sample_params[key+"_anneal"] = True

        batch_size = get_setting("agents") * get_setting("n_steps")

        # convert negative max_horizon to auto
        if sample_params.get("tvf_max_horizon", 0) < 0:
            sample_params["tvf_max_horizon"] = 30000
            sample_params["auto_horizon"] = True

        # make sure mini_batch_size is not larger than batch_size
        if "policy_mini_batch_size" in sample_params:
            sample_params["policy_mini_batch_size"] = min(batch_size, sample_params["policy_mini_batch_size"])
        if "value_mini_batch_size" in sample_params:
            sample_params["value_mini_batch_size"] = min(batch_size, sample_params["value_mini_batch_size"])

        # post-processing hook
        if hook is not None:
            hook(sample_params)

        for j in range(len(envs)):
            env_name = envs[j]
            main_params['env_name'] = env_name
            add_job(
                run,
                run_name=f"{i:04d}_{env_name}",
                chunk_size=10,
                score_threshold=score_thresholds[j] if score_thresholds is not None else None,
                default_params=main_params,
                priority=priority,
                **sample_params,
            )