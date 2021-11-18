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
ATARI_2 = ['Krull', 'KungFuMaster']
DIVERSE_5 = ['BattleZone', 'Gopher', 'TimePilot', "Seaquest", "Breakout"] # not the real atari 5...?


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


def setup_TVF_Extra():

    default_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'architecture': 'dual',
        'export_video': False,
        'epochs': 50,  # really want to see where these go...
        'use_compression': 'auto',
        'warmup_period': 1000,  # helps on some games to make sure they are really out of sync at the beginning of training.
        'disable_ev': False,
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
        'gamma': 0.99997,
        'tvf_gamma': 0.99997,
        'tvf_max_horizon': 30000,
    }

    # this tests if there is any improvement if we skip small updates.
    for threshold in [0, 0.25, 0.5, 0.75, 1.0]:
        add_job(
            f"Extra",
            env_name="Krull",
            run_name=f"threshold_{threshold}",
            use_tvf=False, # want to know how this works without TVF as TVF is more noise resistant.
            default_params=default_args,
            optimizer = "GBOGH",
            gbogh_threshold = threshold,
            priority=0,
            epochs=30,
            hostname='desktop',
        )

    # this tests impact of gae_lambda
    for gae_lambda in [0.9, 0.95, 0.97, 0.99, 1.0]:
        add_job(
            f"Extra",
            env_name="Krull",
            run_name=f"gae_lambda={gae_lambda}",
            default_params=default_args,
            gae_lambda=gae_lambda,
            priority=0,
            hostname='desktop',
        )

    # this tests how robust algorithm is to reward noise.
    for use_tvf in [True, False]:
        for per_step_reward_noise in [0.0, 0.1, 0.5, 1.0, 2.0, 4.0]:
            add_job(
                f"Extra",
                env_name="Krull",
                run_name=f"use_tvf={use_tvf} per_step_reward_noise={per_step_reward_noise}",
                use_tvf=use_tvf,
                per_step_reward_noise=per_step_reward_noise,
                default_params=default_args,
                epochs=30,
                priority=50,
                hostname='desktop',
            )

    # see how we do with deferred rewards
    for horizon in [1000, 5000, 20000]:
        for use_tvf in [True, False]:
            add_job(
                f"Deferred",
                env_name="Krull",
                run_name=f"use_tvf={use_tvf} horizon={horizon}",
                use_tvf=use_tvf,
                timeout=horizon*4,
                deferred_rewards=-1,  # reward given at end of episode
                reward_clipping="off",
                default_params=default_args,
                priority=100,
                hostname='',
            )
        add_job(
            f"Deferred",
            env_name="Krull",
            run_name=f"use_tvf=auto horizon={horizon}",
            use_tvf=True,
            timeout=horizon * 4,
            tvf_max_horizon=horizon,
            reward_clipping="off",
            deferred_rewards=-1,  # reward given at end of episode
            default_params=default_args,
            priority=100,
            hostname='',
        )

    # check how no discounting works...
    for env in ["Krull", "Breakout"]:
        for use_tvf in [True, False]:
            for gamma in [0.99, 1.0]:
                add_job(
                    f"Gamma_{env}",
                    env_name=env,
                    run_name=f"use_tvf={use_tvf} gamma={gamma}",
                    use_tvf=use_tvf,
                    gamma=gamma,
                    tvf_gamma=gamma,
                    tvf_max_horizon=30000 if gamma == 1.0 else round(3/(1-gamma)),
                    default_params=default_args,
                    epochs=20,
                    priority=50,
                    hostname='',
                )

                add_job(
                    f"Gamma_{env}",
                    env_name=env,
                    run_name=f"use_tvf={use_tvf} gamma={gamma} (simple)",
                    use_tvf=use_tvf,
                    gamma=gamma,
                    tvf_gamma=gamma,
                    reward_clipping="off",
                    value_transform="identity",
                    terminal_on_loss_of_life=False,
                    tvf_max_horizon=30000 if gamma == 1.0 else round(3 / (1 - gamma)),
                    default_params=default_args,
                    epochs=20,
                    priority=250,
                    hostname='',
                )

        add_job(
            f"Gamma_{env}",
            env_name=env,
            run_name=f"use_tvf=True gamma=1.0 tvf_gamma=0.99",
            use_tvf=True,
            gamma=1.0,
            tvf_gamma=0.99,
            tvf_max_horizon=300,
            default_params=default_args,
            epochs=20,
            priority=200,
            hostname='',
        )

        add_job(
            f"Gamma_{env}",
            env_name=env,
            run_name=f"use_tvf=True gamma=0.99 tvf_gamma=1.0",
            use_tvf=True,
            gamma=0.99,
            tvf_gamma=1.0,
            tvf_max_horizon=300,
            default_params=default_args,
            epochs=20,
            priority=200,
            hostname='',
        )

    # check if increasing gamma over time helps
    for auto_gamma in ["gamma", "tvf", "off", "both"]:
        add_job(
            f"DynamicGamma",
            env_name="Krull",
            run_name=f"(agent_age) auto_gamma={auto_gamma}",
            use_tvf=True,
            auto_gamma=auto_gamma,
            auto_strategy="agent_age_slow",
            default_params=default_args,
            priority=50,
            hostname='desktop',
        )

        add_job(
            f"DynamicGamma",
            env_name="Krull",
            run_name=f"(ep_length) auto_gamma={auto_gamma}",
            use_tvf=True,
            auto_gamma=auto_gamma,
            auto_strategy="episode_length",
            default_params=default_args,
            priority=50,
            hostname='desktop',
        )

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
        'warmup_period': 1000,  # helps on some games to make sure they are really out of sync at the beginning of training.
        'disable_ev': False,
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
            priority=0,
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
            priority=0,
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
                priority=0,
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
        'warmup_period': 1000,  # helps on some games to make sure they are really out of sync at the beginning of training.
        'disable_ev': False,
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

    # check why pong isn't working...
    for seed in [0, 1, 2]:
        add_job(
            f"Pong",
            env_name="Pong",
            run_name=f"seed={seed}",
            default_params=default_args,
            seed=seed,
            priority=100,
            epochs=20,
            hostname='ML-Rig',
        )

    add_job(
        f"Pong",
        env_name="Pong",
        run_name=f"value_transform=identity",
        default_params=default_args,
        value_transform='identity',
        priority=100,
        epochs=20,
        hostname='ML-Rig',
    )

    add_job(
        f"Pong",
        env_name="Pong",
        run_name=f"use_tvf=False",
        default_params=default_args,
        use_tvf=False,
        priority=100,
        epochs=20,
        hostname='ML-Rig',
    )

    # try pong with different hyperparameters...

    add_job(
        f"Pong",
        env_name="Pong",
        run_name=f"split=1",
        default_params=default_args,

        use_tvf=False,

        # these parameters are taken from DNA, I want to see which hyperparameter makes pong not work.
        max_grad_norm=5.0,
        agents=128,
        n_steps=256,
        policy_mini_batch_size=256,
        value_mini_batch_size=256,
        policy_epochs=4,
        value_epochs=2,
        distill_epochs=3,
        distill_beta=0.5,
        # target_kl=0.03,
        # ppo_epsilon=0.2,
        # policy_lr=1e-4,
        # value_lr=2.5e-4,
        # distill_lr=1e-4,
        # entropy_bonus=1e-3,
        # tvf_force_ext_value_distill=True,
        # hidden_units=128,
        # value_transform='sqrt',
        # gae_lambda=0.95,

        priority=100,
        epochs=20,
        hostname='ML-Rig',
    )

    add_job(
        f"Pong",
        env_name="Pong",
        run_name=f"split=2",
        default_params=default_args,

        use_tvf=False,

        # these parameters are taken from DNA, I want to see which hyperparameter makes pong not work.
        # max_grad_norm=5.0,
        # agents=128,
        # n_steps=256,
        # policy_mini_batch_size=256,
        # value_mini_batch_size=256,
        # policy_epochs=4,
        # value_epochs=2,
        # distill_epochs=3,
        # distill_beta=0.5,

        # one of the following hyperparameters is needed for pong to train.
        target_kl=0.03,
        ppo_epsilon=0.2,
        policy_lr=1e-4,
        value_lr=2.5e-4,
        distill_lr=1e-4,
        entropy_bonus=1e-3,
        tvf_force_ext_value_distill=True,
        hidden_units=128,
        value_transform='sqrt',
        gae_lambda=0.95,

        priority=100,
        epochs=20,
        hostname='ML-Rig',
    )

    add_job(
        f"Pong",
        env_name="Pong",
        run_name=f"split=3a",
        default_params=default_args,

        use_tvf=False,

        # these parameters are taken from DNA, I want to see which hyperparameter makes pong not work.
        # max_grad_norm=5.0,
        # agents=128,
        # n_steps=256,
        # policy_mini_batch_size=256,
        # value_mini_batch_size=256,
        # policy_epochs=4,
        # value_epochs=2,
        # distill_epochs=3,
        # distill_beta=0.5,

        # target_kl=0.03,
        # ppo_epsilon=0.2,
        # policy_lr=1e-4,
        # value_lr=2.5e-4,
        # distill_lr=1e-4,
        entropy_bonus=1e-3,
        tvf_force_ext_value_distill=True,
        hidden_units=128,
        value_transform='sqrt',
        gae_lambda=0.95,

        priority=100,
        epochs=20,
        hostname='ML-Rig',
    )

    add_job(
        f"Pong",
        env_name="Pong",
        run_name=f"split=3b",
        default_params=default_args,

        use_tvf=False,

        # these parameters are taken from DNA, I want to see which hyperparameter makes pong not work.
        # max_grad_norm=5.0,
        # agents=128,
        # n_steps=256,
        # policy_mini_batch_size=256,
        # value_mini_batch_size=256,
        # policy_epochs=4,
        # value_epochs=2,
        # distill_epochs=3,
        # distill_beta=0.5,

        target_kl=0.03,
        ppo_epsilon=0.2,
        policy_lr=1e-4,
        value_lr=2.5e-4,
        distill_lr=1e-4,
        # entropy_bonus=1e-3,
        # tvf_force_ext_value_distill=True,
        # hidden_units=128,
        # value_transform='sqrt',
        # gae_lambda=0.95,

        priority=100,
        epochs=20,
        hostname='ML-Rig',
    )

    add_job(
        f"Pong",
        env_name="Pong",
        run_name=f"split=4a",
        default_params=default_args,

        use_tvf=False,



        # target_kl=0.03,
        # ppo_epsilon=0.2,
        policy_lr=1e-4,
        value_lr=2.5e-4,
        distill_lr=1e-4,


        priority=100,
        epochs=20,
        hostname='ML-Rig',
    )

    add_job(
        f"Pong",
        env_name="Pong",
        run_name=f"split=4b",
        default_params=default_args,

        use_tvf=False,

        target_kl=0.03,
        ppo_epsilon=0.2,
        # policy_lr=1e-4,
        # value_lr=2.5e-4,
        # distill_lr=1e-4,

        priority=100,
        epochs=20,
        hostname='ML-Rig',
    )

    add_job(
        f"Pong",
        env_name="Pong",
        run_name=f"split=5a",
        default_params=default_args,

        use_tvf=False,

        #target_kl=0.03,
        ppo_epsilon=0.2,

        priority=100,
        epochs=20,
        hostname='ML-Rig',
    )

    add_job(
        f"Pong",
        env_name="Pong",
        run_name=f"split=5b",
        default_params=default_args,

        use_tvf=False,

        target_kl=0.03,
        #ppo_epsilon=0.2,

        priority=100,
        epochs=20,
        hostname='ML-Rig',
    )

    # parameters found by hyperparameter search...



# ---------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    id = 0
    job_list = []

    setup_DNA_Atari57()
    setup_TVF_Atari57()
    setup_TVF_Extra()
    setup_ED()

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
