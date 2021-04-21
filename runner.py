import os
import sys
import json
import time
import random
import platform

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

# initial long horizon test... :)
better_args = {
    'checkpoint_every': int(5e6),
    'workers': WORKERS,
    'env_name': 'DemonAttack',
    'epochs': 50,
    'max_grad_norm': 5,             # more than before  [mode=5]
    'agents': 128,                  #                   [mode=128]
    'n_steps': 32,                  # less than before  [mode=128] (higher better)
    'policy_mini_batch_size': 1024, # more than before  [mode=1024]
    'value_mini_batch_size': 512,   #                   [mode=512] (lower better)
    'value_epochs': 4,              # more than before  [mode=2]
    'policy_epochs': 4,             #                   [mode=4] (higher better)
    'target_kl': 0.10,              # much more than before (see if this is an issue...) [no mode]
    'ppo_epsilon': 0.05,            # why so low?       [mode=0.05] (lower is better, but 0.3 works too?)
    'value_lr': 1e-4,               # much lower        [no mode]
    'policy_lr': 5e-4,              # higher? weird...  [mode=2.5e-4]
    'gamma': 0.999,
}

tvf_tuned_adv_args = {
    'use_tvf': True,
    'tvf_hidden_units': 128,
    'tvf_n_horizons': 128,
    'tvf_lambda': 1.0,
    'tvf_coef': 0.01,
    'tvf_max_horizon': 3000,
    'gamma': 0.999,
    'tvf_gamma': 0.999,
    'tvf_loss_weighting': "advanced",
    'tvf_h_scale': "squared",
    **tuned_args
}


# inspired mostly from new PPO search, but only to 10m...
# these did not work...
new_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'env_name': 'DemonAttack',
        'epochs': 50,
        'max_grad_norm': 5,
        'agents': 256,                      # more is probably better
        'n_steps': 128,                     # hard to know, but atleast we are now free the MC algorithm
        'policy_mini_batch_size': 1024,
        'value_mini_batch_size': 256,       # smaller is probably better
        'value_epochs': 2,                  # I probably want 4 with early stopping
        'policy_epochs': 4,
        'target_kl': 0.01,                  # need to search more around this
        'ppo_epsilon': 0.1,                 # hard to say if this is right...
        'value_lr': 2.5e-4,                 # slow and steady wins the race
        'policy_lr': 2.5e-4,

        'use_tvf': True,
        'tvf_hidden_units': 128,            # big guess here
        'tvf_value_samples': 128,           # big guess here
        'tvf_horizon_samples': 32,          # 32 has been shown to be enough
        'tvf_lambda': -16,                  # n-steps is great
        'tvf_coef': 0.01,                   # big guess
        'tvf_max_horizon': 3000,            # could be much higher if we wanted
        'gamma': 0.999,
        'tvf_gamma': 0.999,                 # rediscounting is probably a good idea...
        'tvf_loss_weighting': "advanced",   # these seem to help
        'tvf_h_scale': "squared",
    }


# this are the 'fast but good' settings.
optimized_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'env_name': 'DemonAttack',
        'epochs': 50,
        'max_grad_norm': 5.0,
        'agents': 256,                      # more is probably better
        'n_steps': 128,                     # hard to know, but atleast we are now free the MC algorithm
        'policy_mini_batch_size': 1024,
        'value_mini_batch_size': 256,       # smaller is probably better
        'value_epochs': 2,                  # I probably want 4 with early stopping
        'policy_epochs': 4,
        'target_kl': 0.01,                  # need to search more around this
        'ppo_epsilon': 0.1,                 # hard to say if this is right...
        'value_lr': 2.5e-4,                 # slow and steady wins the race
        'policy_lr': 2.5e-4,

        'use_tvf': True,
        'tvf_hidden_units': 128,            # big guess here
        'tvf_value_samples': 128,           # reducing from 128 samples to 64 is fine, even 16 would work.
        'tvf_horizon_samples': 32,          # 32 has been shown to be better than 128
        'tvf_return_mixing': 1,             # just makes things slowe to use more than 1
        'tvf_lambda': -16,
        'tvf_lambda_samples': 32,           # being cautious here...
        'tvf_coef': 0.01,                   # big guess
        'tvf_max_horizon': 3000,            # could be much higher if we wanted
        'gamma': 0.999,
        'tvf_gamma': 0.999,                 # rediscounting is probably a good idea...
        'tvf_loss_weighting': "advanced",   # these seem to help
        'tvf_h_scale': "squared",
    }

# tweaked optimized settings with constant h_scale
v13_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'epochs': 50,
        'max_grad_norm': 20.0,
        'agents': 256,                      # more is probably better
        'n_steps': 128,                     # hard to know, but at least we are now free the MC algorithm
        'policy_mini_batch_size': 1024,
        'value_mini_batch_size': 256,       # smaller is probably better
        'value_epochs': 4,                  # I probably want 4 with early stopping
        'policy_epochs': 4,
        'target_kl': 0.01,                  # need to search more around this
        'ppo_epsilon': 0.1,                 # hard to say if this is right...
        'value_lr': 2.5e-4,                 # slow and steady wins the race
        'policy_lr': 2.5e-4,

        'use_tvf': True,
        'tvf_hidden_units': 128,            # big guess here
        'tvf_value_samples': 128,           # reducing from 128 samples to 64 is fine, even 16 would work.
        'tvf_horizon_samples': 64,          # 32 has been shown to be better than 128
        'tvf_lambda': -16,
        'tvf_lambda_samples': 32,
        'tvf_coef': 0.1,
        'tvf_max_horizon': 3000,
        'gamma': 0.999,
        'tvf_gamma': 0.999,
        'tvf_loss_weighting': "advanced",
        'tvf_h_scale': "constant",
    }



# this are based on the best model from PPO HPS at 10M
best10_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'epochs': 50,
        'max_grad_norm': 20.0,
        'agents': 64,
        'n_steps': 256,
        'policy_mini_batch_size': 512,
        'value_mini_batch_size': 512,
        'value_epochs': 6,
        'policy_epochs': 4,
        'target_kl': 0.1,
        'ppo_epsilon': 0.1,                 # hard to say if this is right...
        'value_lr': 2.5e-4,                 # value should be 5e-4... but I don't want to do it for TVF
        'policy_lr': 2.5e-4,

        'use_tvf': True,
        'tvf_hidden_units': 128,
        'tvf_value_samples': 128,
        'tvf_horizon_samples': 32,
        'tvf_return_mixing': 1,
        'tvf_lambda': -16,
        'tvf_lambda_samples': 64,
        'tvf_coef': 0.01,
        'tvf_max_horizon': 3000,
        'gamma': 0.999,
        'tvf_gamma': 0.999,
        'tvf_loss_weighting': "advanced",
        'tvf_h_scale': "squared",
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
    print("-" * 161)
    print("{:^10}{:<20}{:<60}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}".format("priority", "experiment_name", "run_name", "complete", "status", "eta", "fps", "score", "host", "ping"))
    print("-" * 161)
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
            host = details["host"][:8] if status == "running" else ""
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

        print("{:^10}{:<20}{:<60}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
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


def random_search(run:str, main_params:dict, search_params:dict, envs:list, score_thresholds:list, count:int=128):

    assert len(envs) == len(score_thresholds)

    for i in range(count):
        params = {}
        random.seed(i)
        for k, v in search_params.items():
            params[k] = random.choice(v)

        # make sure params aren't too high (due to memory)
        while params["agents"] * params["n_steps"] > 64*1024:
            params["agents"] //= 2

        while params["agents"] * params["n_steps"] < params["policy_mini_batch_size"]:
            params["policy_mini_batch_size"] //= 2

        while params["agents"] * params["n_steps"] < params["value_mini_batch_size"]:
            params["value_mini_batch_size"] //= 2

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

def setup_experiments_11():

    add_job(
        f"TVF_11_Regression",
        run_name=f"ppo (tuned)",
        default_params=tuned_args,
        priority=100,
    )

    add_job(
        f"TVF_11_Regression",
        run_name=f"ppo (better)",
        default_params=better_args,
        priority=100,
    )

    add_job(
        f"TVF_11_Regression",
        run_name=f"tvf (tuned)",

        use_tvf=True,
        tvf_hidden_units=128,
        tvf_n_horizons=128,
        tvf_lambda=1.0,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        default_params=tuned_args,
        priority=100,
    )

    add_job(
        f"TVF_11_Regression",
        run_name=f"tvf (better)",

        use_tvf=True,
        tvf_hidden_units=128,
        tvf_n_horizons=128,
        tvf_lambda=1.0,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        default_params=better_args,
        priority=100,
    )

    add_job(
        f"TVF_11_Regression",
        run_name=f"tvf (tuned_adv)",

        use_tvf=True,
        tvf_hidden_units=128,
        tvf_n_horizons=128,
        tvf_lambda=1.0,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=100,
    )

    add_job(
        f"TVF_11_Regression",
        run_name=f"tvf_16 (tuned_adv)",

        use_tvf=True,
        tvf_hidden_units=128,
        tvf_n_horizons=128,
        tvf_lambda=-16,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=100,
    )

    add_job(
        f"TVF_11_Regression",
        run_name=f"tvf (better_adv)",

        use_tvf=True,
        tvf_hidden_units=128,  # mostly a performance optimization
        tvf_n_horizons=128,
        tvf_lambda=1.0,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=better_args,
        priority=100,
    )

    for chunk_size in [5, 10, 20, 50]:
        add_job(
            f"TVF_11_Chunking",
            run_name=f"chunk={chunk_size}",
            epochs=50,
            chunk_size=chunk_size,

            default_params=tvf_tuned_adv_args,
            priority=-100,
        )

    for tvf_n_horizons in [2, 4, 8, 16, 32, 64, 128]:
        add_job(
            f"TVF_11_n_horizons",
            run_name=f"tvf_n_horizons={tvf_n_horizons}",
            epochs=50,
            tvf_n_horizons=tvf_n_horizons,
            default_params=tvf_tuned_adv_args,
            priority=20,
        )


def setup_experiments_11_eval():

    for env_name in ["Alien", "BankHeist", "CrazyClimber"]:
        add_job(
            f"TVF_11_Test_PPO_tuned",
            run_name=f"{env_name}",
            env_name=env_name,
            use_tvf=False,
            epochs=50,
            default_params=tuned_args,
            priority=0,
        )

        add_job(
            f"TVF_11_Test_TVF_tuned_adv",
            run_name=f"{env_name}",
            env_name=env_name,
            epochs=50,

            use_tvf=True,
            tvf_hidden_units=128,  # mostly a performance optimization
            tvf_n_horizons=128,
            tvf_lambda=1.0,
            tvf_coef=0.01,
            tvf_max_horizon=3000,
            gamma=0.999,
            tvf_gamma=0.999,

            tvf_loss_weighting="advanced",
            tvf_h_scale="squared",

            default_params=tuned_args,
            priority=0,
        )

        add_job(
            f"TVF_11_Test_TVF_16_tuned_adv",
            run_name=f"{env_name}",
            env_name=env_name,
            epochs=50,

            use_tvf=True,
            tvf_hidden_units=128,
            tvf_n_horizons=128,
            tvf_lambda=-16,
            tvf_coef=0.01,
            tvf_max_horizon=3000,
            gamma=0.999,
            tvf_gamma=0.999,

            tvf_loss_weighting="advanced",
            tvf_h_scale="squared",

            default_params=tuned_args,
            priority=0,
        )


def random_search_11_ppo():
    main_params = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'export_video': False, # save some space...
        'use_tvf': False,
        'gamma': 0.999,
        'epochs': 30,
        'priority': -20,
    }

    search_params = {
        'max_grad_norm': [0.5, 5, 20.0],    # should have little to no effect
        'agents': [64, 128, 256, 512],      # I expect more is better
        'n_steps': [16, 32, 64, 128, 256],  # I expect more is better
        'policy_mini_batch_size': [512, 1024, 2048],
        'value_mini_batch_size': [256, 512, 1024, 2048], #I expect lower is better
        'value_epochs': [1, 2, 4, 6],
        'policy_epochs': [1, 2, 4, 6],
        'target_kl': [0.01, 0.03, 0.1, 1.0],  # 1.0 is effectively off
        'ppo_epsilon': [0.05, 0.1, 0.2, 0.3], # I have only gotten <= 0.1 to work so far
        'value_lr': [1e-4, 2.5e-4, 5e-4],
        'policy_lr': [1e-4, 2.5e-4, 5e-4],
        'vf_coef': [0.25, 0.5, 1.0],          # I don't think this matters?
        'entropy_bonus': [0.003, 0.01, 0.03]  # probably will not make much difference
    }

    # score threshold should be 200, but I want some early good results...
    random_search(
        "TVF_11_Search_PPO",
        main_params,
        search_params,
        count=32,
        envs=['BattleZone', 'DemonAttack', 'Amidar'],
        score_thresholds=[5000, 500, 50],
    )


def setup_old_regression_experiments():
    add_job(
        f"TVF_12_Regression",
        run_name=f"tvf_16",

        use_tvf=True,
        tvf_hidden_units=128,

        tvf_horizon_samples=32,
        tvf_value_samples=32,

        tvf_lambda=-16,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=200,
    )

    add_job(
        f"TVF_12_Regression",
        run_name=f"tvf_16_v2",

        use_tvf=True,
        tvf_hidden_units=128,

        tvf_horizon_samples=128,  # should be the same as previously
        tvf_value_samples=-1,

        tvf_lambda=-16,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=200,
    )

    add_job(
        f"TVF_12_Regression2",
        run_name=f"tvf_16_v3",

        use_tvf=True,
        tvf_hidden_units=128,

        tvf_horizon_samples=64,
        tvf_value_samples=64,

        tvf_lambda=-16,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=200,
    )

    add_job(
        f"TVF_12_Regression3",
        run_name=f"tvf_16_v4",

        use_tvf=True,
        tvf_hidden_units=128,

        tvf_value_samples=128,
        tvf_horizon_samples=128,

        tvf_lambda=-16,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=200,
    )

    # reset kl back to approx kl... (which is what we tuned for)

    add_job(
        f"TVF_12_Regression5",
        run_name=f"tvf_16_v5",

        use_tvf=True,
        tvf_hidden_units=128,

        tvf_value_samples=128,
        tvf_horizon_samples=128,

        tvf_lambda=-16,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=200,
    )

    add_job(
        f"TVF_12_Regression6",
        run_name=f"tvf_16_v1_dup",

        use_tvf=True,
        tvf_hidden_units=128,

        tvf_horizon_samples=32,
        tvf_value_samples=32,

        tvf_lambda=-16,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=200,
    )

    add_job(
        f"TVF_12_Regression6",
        run_name=f"tvf_16_v6",

        use_tvf=True,
        tvf_hidden_units=128,

        tvf_value_samples=128,
        tvf_horizon_samples=128,

        tvf_lambda=-16,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=200,
    )

    add_job(
        f"TVF_12_Regression6",
        run_name=f"tvf_16_v2_dup",

        use_tvf=True,
        tvf_hidden_units=128,

        tvf_horizon_samples=128,  # should be the same as previously
        tvf_value_samples=-1,

        tvf_lambda=-16,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=200,
    )

    add_job(
        f"TVF_12_Regression7",
        run_name=f"tvf_16_v7",

        use_tvf=True,
        tvf_hidden_units=128,

        value_lr=2.5e-4,

        tvf_value_samples=128,
        tvf_horizon_samples=128,

        tvf_lambda=-16,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=200,
    )

    add_job(
        f"TVF_12_Regression8",
        run_name=f"tvf_16_v8",

        use_tvf=True,
        tvf_hidden_units=128,

        tvf_horizon_samples=128,
        tvf_value_samples=128,

        tvf_lambda=-16,
        tvf_coef=0.01,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,

        # required due to bug fix
        value_lr=2.5e-4,

        tvf_loss_weighting="advanced",
        tvf_h_scale="squared",

        default_params=tuned_args,
        priority=200,
    )


def setup_experiments_13():

    for tvf_h_scale in ['constant', 'linear', 'squared']:
        add_job(
            f"TVF_13_Regression",
            env_name="DemonAttack",
            run_name=f"h_scale={tvf_h_scale}",
            tvf_h_scale=tvf_h_scale,
            default_params=v13_args,
            epochs=50,
            priority=250,
        )

    for gamma in [0.9, 0.99, 0.999, 0.9999, 1.0]:
        for timeout in [100*4, 1000*4]:
            add_job(
                f"TVF_13_TimeLimited",
                env_name="DemonAttack",
                timeout=timeout,
                run_name=f"timeout={timeout} gamma={gamma}",
                tvf_gamma=gamma,
                gamma=gamma,
                default_params=v13_args,
                epochs=20,
                priority=50,
            )


def setup_experiments_12():

    # for tvf_value_distribution in ["constant", "linear", "hyperbolic", "exponential"]:
    #     for tvf_value_samples in [4, 8, 16, 64, 256]:
    #         add_job(
    #             f"TVF_12_ValueSamples_{tvf_value_distribution.capitalize()}",
    #             run_name=f"samples={tvf_value_samples}",
    #             tvf_value_samples=tvf_value_samples,
    #             tvf_value_distribution=tvf_value_distribution,
    #             default_params=new_args,
    #             epochs=50 if tvf_value_distribution == "constant" else 30,
    #             priority=50,
    #         )

    # # check return mixing
    # for return_mixing in [1, 2, 4, 8, 16]:
    #     add_job(
    #         f"TVF_12_ReturnMixing",
    #         run_name=f"return_mixing={return_mixing}",
    #         tvf_return_mixing=return_mixing,
    #         tvf_horizon_samples=128,
    #         tvf_value_samples=128,
    #         tvf_value_distribution="constant",
    #         default_params=new_args,
    #         epochs=30,  # just to get an idea for the moment...
    #         priority=50,
    #     )

    # check samples in x/x/ mode with new settings
    for samples in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        add_job(
            f"TVF_12_Sampling",
            run_name=f"samples={samples}",
            tvf_horizon_samples=samples,
            tvf_value_samples=samples,
            tvf_value_distribution="constant",
            default_params=new_args,
            epochs=50,  # just to get an idea for the moment...
            priority=0,
        )

    # for n_step in [1, 2, 4, 8, 16, 32, 64, 128]:
    #     add_job(
    #         f"TVF_12_Step",
    #         run_name=f"n_step={n_step}",
    #         tvf_lambda=-n_step,
    #         tvf_horizon_samples=64,
    #         tvf_value_samples=64,
    #         tvf_value_distribution="constant",
    #         default_params=new_args,
    #         epochs=50,
    #         priority=50,
    #     )

    # for td_lambda in [0.9, 0.95, 0.97]:
    #     for lambda_samples in [-1, 16]:
    #         add_job(
    #             f"TVF_12_Lambda",
    #             run_name=f"lambda={td_lambda} samples={lambda_samples}",
    #             tvf_lambda=td_lambda,
    #             tvf_lambda_samples=lambda_samples,
    #             tvf_horizon_samples=64,
    #             tvf_value_samples=64,
    #             tvf_value_distribution="constant",
    #             default_params=new_args,
    #             epochs=50,
    #             priority=100,
    #         )

    # for tvf_gamma in [0.999, 0.9999, 1.0]:
    #     for gamma in [0.999]:
    #         add_job(
    #             f"TVF_12_Rediscounting",
    #             run_name=f"tvf_gamma={tvf_gamma} gamma={gamma}",
    #             tvf_horizon_samples=64,
    #             tvf_value_samples=64,
    #             tvf_gamma=tvf_gamma,
    #             gamma=gamma,
    #             tvf_return_mixing=1,
    #             tvf_value_distribution="constant",
    #             default_params=new_args,
    #             epochs=50,
    #             priority=200,
    #         )

    # Long horizons :)
    # this will just tell us if long horizons work out of the box,
    # might need to increase all sampling, so maybe use
    # we know 32/32 works so maybe 512/512
    for gamma, horizon in zip([0.99, 0.999, 0.999, 0.9999, 0.99999, 1], [300, 3000, 30000, 30000, 30000, 30000]):
        add_job(
            f"TVF_12_LongHorizon",
            run_name=f"gamma={gamma} horizon={horizon}",
            gamma=gamma,
            tvf_gamma=gamma,
            tvf_max_horizon=horizon,
            default_params=optimized_args,
            export_video=False,  # not needed
            epochs=100,
            priority=200,
        )
    add_job(
        f"TVF_12_LongHorizon",
        run_name=f"gamma={1.0} horizon={30000} (v2)",
        gamma=1.0,
        tvf_gamma=1.0,
        tvf_max_horizon=30000,
        tvf_value_samples=512,
        tvf_horizon_samples=512,
        default_params=optimized_args,
        export_video=False,
        epochs=100,
        priority=0,
    )

    # Retake on sampling
    # Kind of wish I have av_ev here...
    for samples in [32, 64, 128, 256, 512, 1024]:
        add_job(
            f"TVF_12_FixedHorizonSamples",
            run_name=f"samples={samples}",
            tvf_horizon_samples=samples,
            default_params=optimized_args,
            export_video=False,
            epochs=20,
            priority=50,
        )

    # Hidden units
    for hidden_units in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        add_job(
            f"TVF_12_HiddenUnits",
            run_name=f"hidden_units={hidden_units}",
            tvf_hidden_units=hidden_units,
            default_params=optimized_args,
            export_video=False,
            epochs=20,
            priority=20,
        )

    # Hidden units on game where rewards come all at once
    for hidden_units in [16, 64]:
        add_job(
            f"TVF_12_HiddenUnits_Breakout",
            run_name=f"hidden_units={hidden_units}",
            env_name="Breakout",
            tvf_hidden_units=hidden_units,
            default_params=optimized_args,
            export_video=False,
            epochs=20,
            priority=200,
        )

    # Hidden units on game where rewards come all at once
    for first_and_last in [0, 1/128, 2/128, 4/128]:
        add_job(
            f"TVF_12_FirstAndLast",
            run_name=f"f_and_l={first_and_last} h_samples=128",
            tvf_horizon_samples=128,
            tvf_first_and_last=first_and_last,
            default_params=optimized_args,
            export_video=False,
            epochs=30,
            priority=50,
        )

    # # just want to see if this is a problem
    for env_name in ['Breakout', 'DemonAttack']:
        for h_scale in ['constant', 'linear', 'squared']:
            for loss_weighting in ['default', 'advanced']:
                add_job(
                    f"TVF_12_Curve_{env_name}",
                    env_name=env_name,
                    run_name=f"h_scale={h_scale} loss_weighting={loss_weighting}",
                    tvf_h_scale=h_scale,
                    tvf_loss_weighting=loss_weighting,
                    default_params=optimized_args,
                    export_video=False,
                    epochs=30,
                    priority=150,
                )

    # Long horizons take 2
    # really try to make infinite horizon work well
    # try tvf_lambda, and increased samples
    for ed_type in ['none', 'geometric', 'hyperbolic']:
        add_job(
            f"TVF_12_LongerHorizon",
            run_name=f"tvf_lambda=0.97 ed_type={ed_type}",
            gamma=1.0,
            tvf_gamma=1.0,
            tvf_lambda=0.97,
            tvf_lambda_samples=32,
            tvf_max_horizon=30000,
            tvf_value_samples=256,
            tvf_horizon_samples=256,
            ed_type=ed_type,
            ed_gamma=0.9999,
            default_params=optimized_args,
            export_video=False,
            epochs=30,
            priority=20,
        )

    # Make sure timeout isn't broken
    for timeout in [100*4, 1000*4, 10000*4]:
        add_job(
            f"TVF_12_Timeout",
            run_name=f"timeout={timeout}",
            timeout=timeout,
            default_params=optimized_args,
            export_video=False,
            epochs=20,
            priority=200,
        )

    # Make sure timeout isn't broken
    for timeout in [100*4, 1000*4, 10000*4]:
        add_job(
            f"TVF_12_Timeout_Linear",
            run_name=f"timeout={timeout}",
            timeout=timeout,
            default_params=optimized_args,
            tvf_h_scale="constant",
            export_video=False,
            epochs=20,
            priority=250,
        )


def retired_experiments():
    pass
    # decided to use new as much as I could

    # # just trying to get lower samples to work
    # for tvf_value_distribution in ["hyperbolic_100", "hyperbolic_10", "exponential_4"]:
    #     for tvf_value_samples in [4, 8, 16]:
    #         add_job(
    #             f"TVF_12_VS2",
    #             run_name=f"samples={tvf_value_samples} dist={tvf_value_distribution}",
    #             tvf_value_samples=tvf_value_samples,
    #             tvf_value_distribution=tvf_value_distribution,
    #             default_params=v8_args,
    #             epochs=30,  # just to get an idea for the moment...
    #             priority=30,
    #         )

    # check how many value samples are required
    # for tvf_horizon_samples in [4, 8, 16, 32, 64, 256, 1024]:
    #     add_job(
    #         f"TVF_12_HS",
    #         run_name=f"h_samples={tvf_horizon_samples}",
    #         tvf_horizon_samples=tvf_horizon_samples,
    #         tvf_value_samples=256,
    #         tvf_value_distribution="constant",
    #         default_params=v8_args,
    #         epochs=20,  # just to get an idea for the moment...
    #         priority=0,
    #     )

    # check how many value samples are required
    # this uses v8
    # for tvf_value_distribution in ["constant", "linear", "hyperbolic", "exponential"]:
    #     for tvf_value_samples in [4, 8, 16, 64, 256, -1]:
    #         # only want these high ones for constant, they are just to get an idea of what optimal performance looks like.
    #         if tvf_value_samples == -1 and tvf_value_distribution != "constant":
    #             continue
    #         add_job(
    #             f"TVF_12_VS",
    #             run_name=f"samples={tvf_value_samples} dist={tvf_value_distribution}",
    #             tvf_value_samples=tvf_value_samples,
    #             tvf_value_distribution=tvf_value_distribution,
    #             default_params=v8_args,
    #             epochs=30,  # just to get an idea for the moment...
    #             priority=40,
    #         )


if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    id = 0
    job_list = []
    random_search_11_ppo()
    setup_experiments_13()

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