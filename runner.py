import os
import sys
import json
import time
import random
import platform
import math

import socket

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

# these are the standard args I use for most experiments.
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

    job = Job(experiment_name, run_name, priority, chunk_size, kwargs, hostname=hostname)

    if score_threshold is not None and chunk_size > 0:
        job_details = job.get_details()
        if job_details is not None and 'score' in job_details:
            modified_kwargs = kwargs.copy()
            chunks_completed = job_details['completed_epochs'] / chunk_size
            if job_details['score'] < score_threshold * chunks_completed and chunks_completed > 0.75:
                modified_kwargs["epochs"] = chunk_size
            job = Job(experiment_name, run_name, priority, chunk_size, modified_kwargs, hostname=hostname)

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

    def __init__(self, experiment_name, run_name, priority, chunk_size:int, params, hostname=None):
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
        except Exception as e:
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


        python_part = "python \"{}\" {}".format(train_script_path, self.params["env_name"])

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
    if hours < 24:
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
            total_epochs_to_go += job.params["epochs"] - details["completed_epochs"]
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
        score_thresholds: list,
        count: int = 64,
        process_up_to=None,
        base_seed=0,
        hook=None,
        priority_offset=0,

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
        params = {}
        for k, v in search_params.items():
            params[k] = even_dist_samples[k][i]
            if type(v) in [Uniform, LogUniform] and v._force_int:
                params[k] = int(params[k])
            if type(params[k]) in [float, np.float64]:
                params[k] = smart_round_sig(params[k])

        # agents must divide workers (which we assume divides 8)
        params['agents'] = (params['agents'] // 8) * 8

        batch_size = params["agents"] * params["n_steps"]

        # convert negative learning rates to annealing
        for key in ["policy_lr", "value_lr", "distill_lr"]:
            if key in params and params[key] < 0:
                params[key] = -params[key]
                params[key+"_anneal"] = True

        # convert negative max_horizon to auto
        if params.get("tvf_max_horizon", 0) < 0:
            params["tvf_max_horizon"] = 30000
            params["auto_horizon"] = True

        # make sure mini_batch_size is not larger than batch_size
        if "policy_mini_batch_size" in params:
            params["policy_mini_batch_size"] = min(batch_size, params["policy_mini_batch_size"])
        if "value_mini_batch_size" in params:
            params["value_mini_batch_size"] = min(batch_size, params["value_mini_batch_size"])

        # post processing hook
        if hook is not None:
            hook(params)

        for env_name, score_threshold in zip(envs, score_thresholds):

            hostname = "ML" #if env_name == "Krull" else "desktop"
            priority = 50 if env_name == "Krull" else 0

            main_params['priority'] = priority + priority_offset

            main_params['env_name'] = env_name
            add_job(
                run,
                run_name=f"{i:04d}_{env_name}",
                chunk_size=10,
                score_threshold=score_threshold,
                hostname=hostname,
                seed=-1, # no need to make these deterministic
                **main_params, **params)

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


def E11():
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
            # add_job(
            #     f"E1_1_PerGameGamma",
            #     env_name=env,
            #     run_name=f"game={env} tvf_10k (seed={run})",
            #     use_tvf=True,
            #     gamma=0.9999,
            #     tvf_gamma=0.9999,
            #     default_params=standard_args,
            #     epochs=50,
            #     priority=50,
            #     seed=run,
            #     hostname='',
            # )
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

            add_job(
                # the idea here is that a small dc will allow pong to train
                # the replay probably needed, and I can remove it if we want, but it might also help distillation
                # training
                f"E11_PerGameGamma (additional)",
                env_name=env,
                run_name=f"game={env} replay_simple (seed={run})",
                default_params=replay_simple_args,
                epochs=50,
                priority=-50,
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

def E31():

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
        for replay_size in [0, 128 * 128, 128 * 128 * 2]:
            for replay_mode in ['overwrite', 'uniform', 'mixed']:
                if replay_size == 0 and replay_mode != 'overwrite':
                    continue
                for period in [1, 2]:
                    for epochs in [1, 2]:
                        for dc in [True, False]:
                            if replay_size == 0:
                                rp_tag = ' RP0'
                            else:
                                rp_tag = f" RP{replay_size//(128*128)} ({replay_mode})"

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
        for replay_size in [0, 1, 2, 4, 8]:
            add_job(
                f"test_distil_rp_long",
                env_name=env,
                run_name=f"game={env} replay_simple rs={replay_size}",
                replay_size=replay_size * 128 * 128,
                default_params=replay_simple_args,
                priority=200 if replay_size==8 else -20, # just want to start some of these big ones early...
            )

    for env in ['Pong', 'Breakout', 'CrazyClimber']:

        # see if we can get full curve learning working on pong
        add_job(
            # the idea here is that a small dc will allow pong to train
            # the replay probably needed, and I can remove it if we want, but it might also help distillation
            # training
            f"E11_PerGameGamma (additional)",
            env_name=env,
            run_name=f"game={env} replay_full (seed={1})",
            default_params=replay_full_args,
            epochs=50,
            priority=-50,
            seed=1,
            hostname='',
        )

        # find a good dual constraint
        replay_mode = "uniform"
        for replay_size in [0]:
            for period in [1]:
                for epochs in [1]:
                    for dc in [0, 0.1, 0.3, 1.0, 3.0]:
                        if replay_size == 0:
                            rp_tag = ' RP0'
                        else:
                            rp_tag = f" RP{replay_size//(128*128)} ({replay_mode})"
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


# ---------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    # (not needed anymore...)
    # os.environ["MKL_THREADING_LAYER"] = "GNU"

    id = 0
    job_list = []

    E11()
    E31()

    test_distil()

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
        run_next_experiment(filter_jobs=lambda x: experiment_name in x.run_name)
