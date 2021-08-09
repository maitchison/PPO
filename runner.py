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
WORKERS=8

ATARI_VAL = ['Krull', 'KungFuMaster', 'Seaquest']
ATARI_3 = ['BattleZone', 'Gopher', 'TimePilot']


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

v18_args = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'epochs': 50,
    'export_video': False,
    'use_compression': True,

    # PPO args
    'max_grad_norm': 25.0,
    'agents': 512,                          # want this to be higher, but memory constrained...
    'n_steps': 1024,                        # large n_steps might be needed for long horizon?
    'policy_mini_batch_size': 1024,         # slower but better...
    'value_mini_batch_size': 512,
    'policy_epochs': 3,
    'value_epochs': 4,                      # slightly better with 4
    'distill_epochs': 1,
    'target_kl': -1,                        # remove target_kl
    'ppo_epsilon': 0.2,
    'value_lr': 2.5e-4,
    'policy_lr': 2.5e-4,
    'entropy_bonus': 0.01,                  # was 0.01 but this was on old settings so we need to go lower
    'time_aware': True,
    'distill_beta': 1.0,
    'tvf_force_ext_value_distill': True,    # slightly better, slightly faster...

    # TVF args
    'use_tvf': True,
    'tvf_value_distribution': 'fixed_geometric',
    'tvf_horizon_distribution': 'fixed_geometric',
    'tvf_horizon_scale': 'log',
    'tvf_time_scale': 'log',
    'tvf_hidden_units': 256,               # more is probably better, but we need just ok for the moment
    'tvf_value_samples': 128,
    'tvf_horizon_samples': 128,
    'tvf_mode': 'exponential',             # adaptive seems like a good tradeoff
    'tvf_n_step': 32,                      # perhaps experiment with higher later on?
    'tvf_coef': 1.0,                       # this is an important parameter...
    'tvf_soft_anchor': 1.0,
    'tvf_max_horizon': 30000,

    'gamma': 0.99997,
    'tvf_gamma': 0.99997,

 }


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
        count: int = 64,
        process_up_to=None,
        base_seed=0,
        hook=None,

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

            hostname = "ML" if env_name == "Krull" else "desktop"
            priority = 0

            main_params['priority'] = priority

            main_params['env_name'] = env_name
            add_job(
                run, run_name=f"{i:04d}_{env_name}", chunk_size=10, score_threshold=score_threshold,
                hostname=hostname,
                seed=-1, # no need to make these deterministic
                **main_params, **params)



def random_search_PPO():

    main_params = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'use_tvf': False,
        'architecture': 'single',
        'export_video': False, # save some space...
        'epochs': 50,   # really want to see where these go...
        'use_compression': 'auto',
        'priority': 0,
        'max_grad_norm': 25,
        'warmup_period': 1000, # helps on some games to make sure they are really out of sync at the beginning of training.
        'disable_ev': True,    # a bit faster, and not needed for HPS

        # parameters I don't want to search over...
        'target_kl': -1,   # ignore...

        # env parameters
        'time_aware': False,
        'terminal_on_loss_of_life': True, # to match rainbow
        'reward_clipping': 1,  # to match rainbow
        # question: did rainbow use the full action set? they say the used the "number of actions available in the game"
    }

    search_params = {

        # ppo params
        'agents':           Categorical(32, 64, 128, 256, 512),
        'n_steps':          Categorical(32, 64, 128, 256, 512),

        # what we actually got was... (in terms of effective horizon)
        # 30, 100, 300, <missing 1000>, <missing 3000>, <10000>, <inf>
        # so redo with
        # 100, 300, 1000, 3000, 10000, 30000

        'gamma':            Categorical(0.99, 0.997, 0.999, 0.9997, 0.9999, 0.99997),
        'vf_coef':          Categorical(0.25, 0.5, 1.0),

        'policy_mini_batch_size': Categorical(128, 256, 512, 1024),
        'hidden_units':     Categorical(128, 256, 512),

        'policy_epochs':    Categorical(3, 4, 5),
        'ppo_epsilon':      Categorical(0.1, 0.2, 0.3),

        'policy_lr':         Categorical(1e-4, 2.5e-4, 5.0e-4, 1e-3),
        'learning_rate_anneal': Categorical(False, True),
        'entropy_bonus':    Categorical(0.001, 0.003, 0.01, 0.03, 0.1),

    }

    random_search(
        "PPO_SEARCH",
        main_params,
        search_params,
        count=60,
        process_up_to=60,
        envs=['Krull', 'KungFuMaster', 'Seaquest'],
        score_thresholds=[0, 0, 0],
    )

def random_search_TVF():

    # would have been nice to do...
    # include gae 0.9, 0.95, 0.97
    # include sqrt...

    main_params = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'export_video': False, # save some space...
        'epochs': 50,   # really want to see where these go...
        'use_compression': 'auto',
        'priority': 0,
        'warmup_period': 1000, # helps on some games to make sure they are really out of sync at the beginning of training.
        'disable_ev': True,    # a bit faster, and not needed for HPS

        'architecture': 'dual',
        'time_aware': True,
    }

    search_params = {

        # ppo params
        'max_grad_norm':    Categorical(5.0, 25.0),
        'agents':           Categorical(128, 256, 512),
        'n_steps':          Categorical(128, 256, 512, 1024),
        'policy_mini_batch_size': Categorical(256, 512, 1024),
        'value_mini_batch_size': Categorical(256, 512, 1024),
        'policy_epochs':    Categorical(2, 3, 4),
        'value_epochs':     Categorical(2, 3, 4),
        'distill_epochs':   Categorical(0, 1, 2, 3),
        'distill_beta':     Categorical(0.5, 1.0, 2.0),
        'target_kl':        Categorical(-1, 0.01, 0.003, 0.03),
        'ppo_epsilon':      Categorical(0.1, 0.15, 0.2, 0.25, 0.3),
        'policy_lr':        Categorical(1e-4, 2.5e-4, -5.0e-4),
        'value_lr':         Categorical(1e-4, 2.5e-4, -5.0e-4),
        'distill_lr':       Categorical(1e-4, 2.5e-4, -5.0e-4),
        'entropy_bonus':    Categorical(0.001, 0.003, 0.01),
        'tvf_force_ext_value_distill': Categorical(True, False),
        'hidden_units':     Categorical(128, 256, 512),
        'value_transform':  Categorical('identity', 'sqrt'),
        'gae_lambda':       Categorical(0.9, 0.95, 0.97),

        # tvf params
        'use_tvf': Categorical(True, False),
        'tvf_value_distribution': Categorical('fixed_geometric', 'fixed_linear'),
        'tvf_horizon_distribution': Categorical('fixed_geometric', 'fixed_linear'),
        'tvf_horizon_scale': Categorical('log'),
        'tvf_time_scale': Categorical('log'),
        'tvf_hidden_units': Categorical(128, 256, 512),
        'tvf_value_samples': Categorical(32, 64, 128),
        'tvf_horizon_samples': Categorical(32, 64, 128),
        'tvf_mode': Categorical('exponential', 'nstep'),
        'tvf_n_step': Categorical(40, 80, 120),
        'tvf_exp_gamma': Categorical(1.5, 2.0, 2.5),
        'tvf_coef': Categorical(0.5, 1.0, 2.0),
        'tvf_soft_anchor': Categorical(-1, 0, 1.0, 10.0),
        'tvf_exp_mode': Categorical("default", "transformed", "masked"),

        # linked...
        'tvf_max_horizon': Categorical(1000, 10000, 30000, -1),
    }

    def set_gammas(params):
        HORIZONS = [1000, 10000, 30000, -1]
        GAMMAS   = [0.997, 0.9997, 0.99997, 0.99997]
        assert params["tvf_max_horizon"] in HORIZONS
        idx = HORIZONS.index(params["tvf_max_horizon"])
        params["gamma"] = GAMMAS[idx]
        params["tvf_gamma"] = GAMMAS[idx]

    random_search(
        "TVF_SEARCH",
        main_params,
        search_params,
        count=120,
        process_up_to=120,
        envs=['Krull', 'KungFuMaster'],
        score_thresholds=[0, 0],
        hook=set_gammas,
    )

def setup_tests():

    default_args = v18_args.copy()
    default_args['value_lr'] = 1.0e-4
    default_args['tvf_force_ext_value_distill'] = True
    default_args['entropy_bonus'] = 0.01
    default_args['seed'] = 1  # force deterministic
    default_args['use_compression'] = True
    default_args['eb_beta'] = -0.2
    default_args['tvf_exp_masked'] = True
    #default_args['warmup_period'] = 1000 # needed to make time pilot initial score correct (otherwise we get all low scores then all high scores).

    # trying to get to the bottom of the time pilot issues

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"default",
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"tvf_1k",
        tvf_gamma=0.999,
        gamma=0.999,
        tvf_max_horizon=3000,
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"no_eb_decay",
        eb_beta=0,
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"sample_sum",
        tvf_sample_reduction="sum",
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"random_sample",
        tvf_value_distribution="geometric",
        tvf_horizon_distribution="geometric",
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"reward_scale",
        reward_scale=2.0,
        default_params=default_args,
        priority=300,
    )


    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"default (bugfixed)",
        default_params=default_args,
        priority=400,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"no_mask",
        tvf_exp_masked=False,
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"no_compression",
        use_compression=False,
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"lower_entropy",
        entropy_bonus=0.003,
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"very_low_entropy",
        entropy_bonus=0.001,
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"full_distill",
        tvf_force_ext_value_distill=False,
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"no_distill",
        distill_epochs=0,
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"faster_value_lr",
        value_lr=2.5e-4,
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"ppo_1k",
        use_tvf=False,
        gamma=0.999,
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"ppo_10k",
        use_tvf=False,
        gamma=0.9999,
        default_params=default_args,
        priority=300,
    )

    add_job(
        f"PAPER_Test_A",
        env_name="TimePilot",
        run_name=f"no_soft_anchor",
        tvf_soft_anchor=0,
        default_params=default_args,
        priority=300,
    )

    for auto_gamma in ["off", "tvf", "both", "gamma"]:
        add_job(
            f"PAPER_AutoGamma",
            env_name="TimePilot",
            run_name=f"{auto_gamma}",
            auto_gamma=auto_gamma,
            auto_horizon=True,
            default_params=default_args,
            priority=0,
        )

    if True:
        for reward_clipping in ["off", "sqrt", "1"]:
            add_job(
                f"PAPER_RewardClipping",
                env_name="TimePilot",
                run_name=f"reward_clipping={reward_clipping}",
                reward_clipping=reward_clipping,
                default_params=default_args,
                priority=50,
            )


    if True:
        add_job(
            f"PAPER_Test_E",
            env_name="TimePilot",
            run_name=f"implicit_zero",
            tvf_implicit_zero=True,
            default_params=default_args,
            priority=450,
        )

        add_job(
            f"PAPER_Test_F",
            env_name="TimePilot",
            run_name=f"fixed_heads=16",
            tvf_n_dedicated_value_heads=16,
            default_params=default_args,
            priority=300,
        )

        add_job(
            f"PAPER_Test_E",
            env_name="TimePilot",
            run_name=f"no_warmup",
            warmup_period=1,
            default_params=default_args,
            priority=300,
        )

        add_job(
            f"PAPER_Test_E",
            env_name="TimePilot",
            run_name=f"long_warmup",
            warmup_period=2000,
            default_params=default_args,
            priority=300,
        )

    if True:
        add_job(
            f"PAPER_1k",
            env_name="TimePilot",
            run_name=f"gamma=0.999",
            gamma=0.999,
            default_params=default_args,
            priority=0,
        )
        add_job(
            f"PAPER_1k",
            env_name="TimePilot",
            run_name=f"tvf_gamma=0.999",
            tvf_gamma=0.999,
            default_params=default_args,
            priority=0,
        )
        add_job(
            f"PAPER_1k",
            env_name="TimePilot",
            run_name=f"both_gamma=0.999",
            tvf_gamma=0.999,
            gamma=0.999,
            default_params=default_args,
            priority=0,
        )
        add_job(
            f"PAPER_1k",
            env_name="TimePilot",
            run_name=f"tvf_max_horizon=1000",
            tvf_max_horizon=1000,
            default_params=default_args,
            priority=0,
        )
        add_job(
            f"PAPER_1k",
            env_name="TimePilot",
            run_name=f"reward_gamma=0.999",
            override_reward_normalization_gamma=0.999,
            default_params=default_args,
            priority=0,
        )


def setup_paper_experiments2():

    # changes:
    # faster value lr, with less epochs
    # use of 'implicit zero'.

    default_args = v18_args.copy()
    default_args['value_lr'] = 2.5e-4
    default_args['tvf_force_ext_value_distill'] = False
    default_args['entropy_bonus'] = 0.01
    default_args['seed'] = 1  # force deterministic
    default_args['use_compression'] = True
    default_args['eb_beta'] = -0.2
    default_args['tvf_exp_masked'] = False
    default_args['tvf_implicit_zero'] = True
    default_args['tvf_soft_anchor'] = 0
    default_args['value_epochs'] = 2
    default_args['auto_gamma'] = "both"
    default_args['auto_horizon'] = True
    default_args['warmup_period'] = 1000  # needed to make time pilot initial score correct (otherwise we get all low scores then all high scores).


    # check the effect of terminal on loss of life...
    # basically the value function is going to be extremely noisy until the agent learns the number of lives
    # which could take some time.  Way around this is just to train for longer... Alternative solution is to
    # pass number of lives into the agent specifically.
    add_job(
        f"PAPER_TOLL",
        env_name="TimePilot",
        run_name=f"default",
        default_params=default_args,
        priority=400,
        hostname='desktop',
    )
    add_job(
        f"PAPER_TOLL",
        env_name="TimePilot",
        run_name=f"toll",
        default_params=default_args,
        terminal_on_loss_of_life=True,
        priority=400,
        hostname='desktop',
    )
    add_job(
        f"PAPER_TOLL",
        env_name="TimePilot",
        run_name=f"toll_clip",
        default_params=default_args,
        terminal_on_loss_of_life=True,
        reward_clipping=1,
        priority=400,
        hostname='desktop',
    )

    # second attempt...
    # auto horizon will also be adjusted...
    add_job(
        f"PAPER_TOLL2",
        env_name="TimePilot",
        run_name=f"clip",
        default_params=default_args,
        reward_clipping=1,
        priority=300,
        hostname='desktop',
    )
    add_job(
        f"PAPER_TOLL2",
        env_name="TimePilot",
        run_name=f"toll",
        default_params=default_args,
        terminal_on_loss_of_life=True,
        priority=300,
        hostname='desktop',
    )
    add_job(
        f"PAPER_TOLL2",
        env_name="TimePilot",
        run_name=f"toll_clip",
        default_params=default_args,
        terminal_on_loss_of_life=True,
        reward_clipping=1,
        priority=300,
        hostname='desktop',
    )

    for env in ATARI_3:
        add_job(
            f"PAPER_Trial",
            env_name=env,
            run_name=f"{env}_default",
            default_params=default_args,
            priority=100,
            hostname='ML-Rig',
        )
        add_job(
            f"PAPER_Trial",
            env_name=env,
            run_name=f"{env}_lowentropy",
            entropy_bonus=0.003,
            default_params=default_args,
            priority=100,
            hostname='ML-Rig',
        )
        add_job(
            f"PAPER_Trial",
            env_name=env,
            run_name=f"{env}_faster",
            default_params=default_args,
            use_compression=False,
            agents=256,
            n_steps=256,
            priority=150,
            hostname='desktop',
        )


    for run in [1]:
        priority = 100 if run == 1 else 0
        add_job(
            f"PAPER_Trial_Skiing",
            env_name="Skiing",
            run_name=f"default_{run}",
            default_params=default_args,
            priority=priority+50,
            epochs=100,
            seed=run,
            hostname='desktop',
        )
        # add_job(
        #     f"PAPER_Trial_Skiing",
        #     env_name="Skiing",
        #     run_name=f"sticky_{run}" if run >= 2 else "sticky",
        #     default_params=default_args,
        #     sticky_actions=True,
        #     priority=priority,
        #     epochs=100,
        #     seed=run,
        #     hostname='desktop',
        # )
        add_job(
            f"PAPER_Trial_Skiing",
            env_name="Skiing",
            run_name=f"ppo_9999_{run}" if run >= 2 else "ppo_9999",
            default_params=default_args,
            priority=priority,
            use_tvf=False,
            gamma=0.9999,
            auto_gamma="off",
            auto_horizon=False,
            epochs=100,
            seed=run,
            hostname='desktop',
        )
        # add_job(
        #     f"PAPER_Trial_Skiing",
        #     env_name="Skiing",
        #     run_name=f"ppo_999_{run}",
        #     default_params=default_args,
        #     priority=priority,
        #     use_tvf=False,
        #     gamma=0.999,
        #     auto_gamma="off",
        #     auto_horizon=False,
        #     epochs=100,
        #     seed=run,
        #     hostname='desktop',
        # )
        # add_job(
        #     f"PAPER_Trial_Skiing",
        #     env_name="Skiing",
        #     run_name=f"ppo_99_{run}",
        #     default_params=default_args,
        #     priority=priority,
        #     use_tvf=False,
        #     gamma=0.99,
        #     auto_gamma="off",
        #     auto_horizon=False,
        #     epochs=100,
        #     seed=run,
        #     hostname='desktop',
        # )
        # add_job(
        #     f"PAPER_Trial_Skiing",
        #     env_name="Skiing",
        #     run_name=f"tvf_999_{run}" if run >= 2 else "tvf_999",
        #     default_params=default_args,
        #     priority=priority,
        #     tvf_gamma=0.999,
        #     gamma=0.999,
        #     tvf_max_horizon=3000,
        #     auto_gamma="off",
        #     auto_horizon=False,
        #     epochs=100,
        #     seed=run,
        #     hostname='desktop',
        # )

    # add_job(
    #     f"PAPER_Trial_Skiing",
    #     env_name="Skiing",
    #     run_name=f"perfect_control_1",
    #     default_params=default_args,
    #     noop_duration=0,
    #     frame_skip=1,
    #     epochs=400,
    #     priority=300,
    #     eb_beta=-0.2/4, # to account for frameskip
    #     seed=1,
    #     hostname='desktop',
    # )

    for env in canonical_57:

        if env in ["Surround"]:
            # Surround has (known) problem with gym... will have to skip it.
            continue

        # front run these to see how it's going
        if env in ["DemonAttack", "Breakout", "Skiing"]:
            priority = 100
        elif env in ATARI_3 or env in ATARI_VAL:
            priority = 50
        else:
            priority = 0

        add_job(
            f"PAPER_Trial_Atari57",
            env_name=env,
            run_name=f"{env}",
            default_params=default_args,
            priority=priority,
            hostname='ML-Rig',
        )

    # rom is broken for this, so run on my machine...
    add_job(
        f"PAPER_Trial_Atari57",
        env_name="Defender",
        run_name=f"Defender",
        default_params=default_args,
        priority=500,
        hostname='desktop',
    )


def bonus_experiments():
    default_args = v18_args.copy()
    default_args['value_lr'] = 2.5e-4
    default_args['tvf_force_ext_value_distill'] = False
    default_args['entropy_bonus'] = 0.01
    default_args['seed'] = 1  # force deterministic
    default_args['use_compression'] = True
    default_args['eb_beta'] = -0.2
    default_args['tvf_exp_masked'] = False
    default_args['tvf_implicit_zero'] = True
    default_args['tvf_soft_anchor'] = 0
    default_args['value_epochs'] = 2
    default_args['auto_gamma'] = "both"
    default_args['auto_horizon'] = True
    default_args[
        'warmup_period'] = 1000  # needed to make time pilot initial score correct (otherwise we get all low scores then all high scores).

    ATARI_VAL = ['Krull', 'KungFuMaster', 'Seaquest']
    ATARI_3 = ['BattleZone', 'Gopher', 'TimePilot']

    for env in ATARI_3:
        add_job(
            f"BONUS_Impala",
            env_name=env,
            run_name=f"{env}_impala",
            default_params=default_args,
            network="impala",
            priority=500,
            hostname='ML-Rig',
            epochs=10, # just to get a feel for things...
        )


def setup_paper_experiments_sqrt():

    # changes:
    # faster value lr, with less epochs
    # use of 'implicit zero'.

    default_args = v18_args.copy()
    default_args['value_lr'] = 2.5e-4
    default_args['tvf_force_ext_value_distill'] = False
    default_args['entropy_bonus'] = 0.01
    default_args['seed'] = 1  # force deterministic
    default_args['use_compression'] = True
    default_args['eb_beta'] = -0.2
    default_args['tvf_exp_masked'] = False
    default_args['tvf_implicit_zero'] = True
    default_args['tvf_soft_anchor'] = 0
    default_args['value_epochs'] = 2
    default_args['auto_gamma'] = "both"
    default_args['auto_horizon'] = True
    default_args['warmup_period'] = 1000  # needed to make time pilot initial score correct (otherwise we get all low scores then all high scores).

    default_args['hidden_units'] = default_args["tvf_hidden_units"]
    default_args['mode'] = "TVF"
    del default_args['use_tvf']

    del default_args["tvf_hidden_units"]

    ATARI_VAL = ['Krull', 'KungFuMaster', 'Seaquest']
    ATARI_3 = ['BattleZone', 'Gopher', 'TimePilot']

    # # I forget what good results are, but maybe 10k?
    # add_job(
    #     f"PAPER_SQRT",
    #     env_name="TimePilot",
    #     run_name=f"sqrt",
    #     default_params=default_args,
    #     value_transform="sqrt",
    #     priority=350,
    #     hostname='desktop',
    # )
    #
    # # fixed so input to GAE is untransformed
    # add_job(
    #     f"PAPER_SQRT_FIX",
    #     env_name="TimePilot",
    #     run_name=f"sqrt",
    #     default_params=default_args,
    #     value_transform="sqrt",
    #     priority=350,
    #     hostname='desktop',
    # )

    # # scaled transform by 25
    # add_job(
    #     f"PAPER_SQRT_FIX3",
    #     env_name="TimePilot",
    #     run_name=f"sqrt",
    #     default_params=default_args,
    #     value_transform="sqrt",
    #     priority=300,
    #     hostname='desktop',
    # )

    # fixed inverse transform and now has a scale setting
    for reward_scale in [0.1, 1, 10, 25, 100, 1000]:
        add_job(
            f"PAPER_SQRT_FIX4",
            env_name="TimePilot",
            run_name=f"sqrt_{reward_scale}",
            default_params=default_args,
            value_transform="sqrt",
            reward_scale=reward_scale,
            priority=500,
            hostname='desktop',
        )
    add_job(
        f"PAPER_SQRT_FIX4",
        env_name="TimePilot",
        run_name=f"sqrt_1_toll",
        default_params=default_args,
        value_transform="sqrt",
        priority=500,
        terminal_on_loss_of_life=True,
        hostname='desktop',
    )

    add_job(
        f"PAPER_SQRT_FIX4",
        env_name="TimePilot",
        run_name=f"sqrt_1_no_norm",
        default_params=default_args,
        value_transform="sqrt",
        priority=500,
        reward_normalization=False,
        hostname='desktop',
    )

    # add_job(
    #     f"PAPER_SQRT_FIX",
    #     env_name="TimePilot",
    #     run_name=f"sqrt_toll",
    #     default_params=default_args,
    #     value_transform="sqrt",
    #     terminal_on_loss_of_life=True,
    #     priority=200,
    #     hostname='desktop',
    # )
    #
    # add_job(
    #     f"PAPER_SQRT",
    #     env_name="TimePilot",
    #     run_name=f"default",
    #     default_params=default_args,
    #     priority=300,
    #     hostname='desktop',
    # )
    #
    # add_job(
    #     f"PAPER_SQRT",
    #     env_name="TimePilot",
    #     run_name=f"toll_clip",
    #     default_params=default_args,
    #     terminal_on_loss_of_life=True,
    #     reward_clipping=1,
    #     priority=300,
    #     hostname='desktop',
    # )

    # quick EMA test

    for env in ATARI_3:
        add_job(
            f"BONUS_EMA",
            env_name=env,
            run_name=f"{env}_gamma=2.0",
            default_params=default_args,
            ema_frame_stack_gamma=2.0,
            ema_frame_stack=True,
            priority=-500,
        )
        add_job(
            f"BONUS_EMA",
            env_name=env,
            run_name=f"{env}_gamma=2.0_x8",
            default_params=default_args,
            ema_frame_stack_gamma=2.0,
            frame_stack=8,
            ema_frame_stack=True,
            priority=-500,
        )



    add_job(
        f"BONUS_EMA",
        env_name="Seaquest",
        run_name=f"default",
        default_params=default_args,
        priority=500,
        hostname='desktop',
    )

    add_job(
        f"BONUS_EMA",
        env_name="Seaquest",
        run_name=f"ema_gamma=2.0",
        default_params=default_args,
        ema_frame_stack_gamma=2.0,
        ema_frame_stack=True,
        priority=500,
        hostname='desktop',
    )

    add_job(
        f"BONUS_EMA",
        env_name="Seaquest",
        run_name=f"ema_gamma=4.0",
        default_params=default_args,
        ema_frame_stack_gamma=4.0,
        ema_frame_stack=True,
        priority=500,
        hostname='desktop',
    )


def setup_experiments_extra():

    # changes:
    # faster value lr, with less epochs
    # use of 'implicit zero'.

    default_args = v18_args.copy()
    default_args['value_lr'] = 2.5e-4
    default_args['tvf_force_ext_value_distill'] = False
    default_args['entropy_bonus'] = 0.01
    default_args['seed'] = 1  # force deterministic
    default_args['use_compression'] = True
    default_args['eb_beta'] = -0.2
    default_args['tvf_exp_masked'] = False
    default_args['tvf_implicit_zero'] = True
    default_args['tvf_soft_anchor'] = 0
    default_args['value_epochs'] = 2
    default_args['auto_gamma'] = "both"
    default_args['auto_horizon'] = True
    default_args['warmup_period'] = 1000  # needed to make time pilot initial score correct (otherwise we get all low scores then all high scores).

    default_args['hidden_units'] = default_args["tvf_hidden_units"]
    default_args['architecture'] = 'dual'
    default_args['use_tvf'] = True

    del default_args["tvf_hidden_units"]

    ATARI_VAL = ['Krull', 'KungFuMaster', 'Seaquest']
    ATARI_3 = ['BattleZone', 'Gopher', 'TimePilot']

    # note: should redo baseline as rom may have changed..

    # check huber loss, with delta=0 being L1
    # fixed inverse transform and now has a scale setting
    for delta in [1.0]:
        add_job(
            f"EXTRA",
            env_name="TimePilot",
            run_name=f"huber_{delta}",
            default_params=default_args,
            use_huber_loss=True,
            huber_loss_delta=delta,
            priority=200,
            hostname='desktop',
        )

    add_job(
        f"EXTRA",
        env_name="TimePilot",
        run_name=f"tanh_0.2",  # I think epsilon is 0.2?
        default_params=default_args,
        use_tanh_clipping=True,
        priority=200,
        hostname='desktop',
    )

def setup_LO_tests():
    default_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'use_tvf': False,
        'architecture': 'single',
        'export_video': False,  # save some space...
        'epochs': 50,  # really want to see where these go...
        'use_compression': 'auto',
        'max_grad_norm': 25,
        'warmup_period': 1000,
        # helps on some games to make sure they are really out of sync at the beginning of training.
        'disable_ev': False,
        'seed': 0,

        # parameters I don't want to search over...
        'target_kl': -1,  # ignore...

        # env parameters
        'time_aware': False,
        'terminal_on_loss_of_life': True,  # to match rainbow
        'reward_clipping': 1,  # to match rainbow

        # ppo params
        'agents': 256,
        'n_steps': 64,

        'gamma': 0.997,
        'vf_coef': 1.0,

        'policy_mini_batch_size': 256,
        'hidden_units': 512,  # should be 128, but stick to this as we want to use the same model as rainbow-dqn

        'policy_epochs': 5,
        'ppo_epsilon': 0.1,  # less might be better.

        'policy_lr': 1.0e-4,
        'entropy_bonus': 0.01,  # not optimal, but close
    }
    for alpha in [-1, -0.1, 0, 0.1, 1]:
        add_job(
            f"PPO_LogOptimal",
            env_name="TimePilot",
            run_name=f"alpha={alpha}",
            default_params=default_args,
            use_log_optimal=True,
            lo_alpha=alpha,
            priority=100,
            hostname='ML-Rig',
        )
    add_job(
        f"PPO_LogOptimal",
        env_name="TimePilot",
        run_name=f"reference",
        default_params=default_args,
        use_log_optimal=False,
        priority=100,
        hostname='ML-Rig',
    )



def setup_PPO_Atari57():

    default_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'use_tvf': False,
        'architecture': 'single',
        'export_video': False,  # save some space...
        'epochs': 50,  # really want to see where these go...
        'use_compression': 'auto',
        'max_grad_norm': 25,
        'warmup_period': 1000,
        # helps on some games to make sure they are really out of sync at the beginning of training.
        'disable_ev': False,
        'seed': 0,

        # parameters I don't want to search over...
        'target_kl': -1,  # ignore...

        # env parameters
        'time_aware': False,
        'terminal_on_loss_of_life': True,  # to match rainbow
        'reward_clipping': 1,  # to match rainbow

        # ppo params
        'agents': 256,
        'n_steps': 64,

        'gamma': 0.997,
        'vf_coef': 1.0,

        'policy_mini_batch_size': 256,
        'hidden_units': 512, # should be 128, but stick to this as we want to use the same model as rainbow-dqn

        'policy_epochs': 5,
        'ppo_epsilon': 0.1, # less might be better.

        'policy_lr': 1.0e-4,
        'learning_rate_anneal': False,
        'entropy_bonus': 0.01, # not optimal, but close
    }

    for env in canonical_57:
        if env in ["Surround"]:
            continue

        # # front run Atari-3 and Atari-Val to see how it's going
        # if env in ATARI_3 or env in ATARI_VAL:
        #     priority = 50
        # else:
        #     priority = 0

        priority = 0

        add_job(
            f"PPO_Atari57_med",
            env_name=env,
            run_name=f"{env}_best",
            default_params=default_args,
            priority=50,
            hostname='ML-Rig',
        )

        add_job(
            f"PPO_Atari57_low",
            env_name=env,
            run_name=f"{env}_low",
            entropy_bonus=0.003,
            default_params=default_args,
            priority=50,
            hostname='ML-Rig',
        )

        add_job(
            f"PPO_Atari57_high",
            env_name=env,
            run_name=f"{env}_high",
            entropy_bonus=0.03,
            default_params=default_args,
            priority=50,
            hostname='ML-Rig',
        )


def setup_PPO_Additional():

    default_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'use_tvf': False,
        'architecture': 'single',
        'export_video': False,  # save some space...
        'epochs': 50,  # really want to see where these go...
        'use_compression': 'auto',
        'max_grad_norm': 25,
        'warmup_period': 1000,
        # helps on some games to make sure they are really out of sync at the beginning of training.
        'disable_ev': False,

        # parameters I don't want to search over...
        'target_kl': -1,  # ignore...

        # env parameters
        'time_aware': False,
        'terminal_on_loss_of_life': True,  # to match rainbow
        'reward_clipping': 1,  # to match rainbow

        # ppo params
        'agents': 256,
        'n_steps': 64,

        'gamma': 0.997,
        'vf_coef': 1.0,

        'policy_mini_batch_size': 256,
        'hidden_units': 512, # should be 128, but stick to this as we want to use the same model as rainbow-dqn

        'policy_epochs': 5,
        'ppo_epsilon': 0.1, # less might be better.

        'policy_lr': 1.0e-4,
        'learning_rate_anneal': False, # does not help
        'entropy_bonus': 0.01, # not optimal, but close
    }

    for run in [1, 2, 3]:
        for env in ATARI_3:
            add_job(
                f"PPO_Atari3",
                env_name=env,
                run_name=f"{env}_sticky_{run}",
                default_params=default_args,
                sticky_actions=True,
                noop_duration=0,
                entropy_bonus=0.01,
                priority=-50,
                hostname='',
                seed=run,
            )

            add_job(
                f"PPO_Atari3",
                env_name=env,
                run_name=f"{env}_default_{run}",
                default_params=default_args,
                entropy_bonus=0.01,
                priority=-50,
                hostname='',
                seed=run,
            )

            add_job(
                f"PPO_Atari3",
                env_name=env,
                run_name=f"{env}_lowentropy_{run}",
                default_params=default_args,
                entropy_bonus=0.003,
                priority=-50,
                hostname='',
                seed=run,
            )

            add_job(
                f"PPO_Atari3",
                env_name=env,
                run_name=f"{env}_highentropy_{run}",
                default_params=default_args,
                entropy_bonus=0.03,
                priority=-50,
                hostname='',
                seed=run,
            )


# ---------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    id = 0
    job_list = []

    setup_PPO_Atari57()
    setup_LO_tests()

    #random_search_PPO()
    random_search_TVF()

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
