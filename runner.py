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

        if "search" in self.experiment_name.lower():
            # with search we want to make sure we complete partial runs first
            priority = priority + self.get_completed_epochs()
        else:
            priority = priority - self.get_completed_epochs()

        # priority = priority - self.get_completed_epochs()

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

def random_search(run, main_params, search_params, envs:list, score_thresholds=list, count=128):

    for i in range(count):
        params = {}
        random.seed(i)
        for k, v in search_params.items():
            params[k] = random.choice(v)

        # make sure params arn't too high (due to memory)
        while params["agents"] * params["n_steps"] > 64*1024:
            params["agents"] //= 2

        while params["agents"] * params["n_steps"] < params["policy_mini_batch_size"]:
            params["policy_mini_batch_size"] //= 2

        while params["agents"] * params["n_steps"] < params["value_mini_batch_size"]:
            params["value_mini_batch_size"] //= 2

        for env_name, score_threshold in zip(envs, score_thresholds):
            main_params['env_name'] = env_name
            add_job(run, run_name=f"{i:04d}_{env_name}", chunked=True, score_threshold=score_threshold, **main_params, **params)


def nice_format(x):
    if type(x) is str:
        return f'"{x}"'
    if x is None:
        return "None"
    if type(x) in [int, float, bool]:
        return str(x)

    return f'"{x}"'

# ---------------------------------------------------------------------------------------------------------

#
# def setup_experiments_10():
#
#     # these are just for the regression test
#     initial_args = {
#         'checkpoint_every': int(5e6),
#         'workers': WORKERS,
#         'env_name': 'DemonAttack',
#         'epochs': 50,
#         'agents': 256,
#         'n_steps': 256,
#         'max_grad_norm': 5.0,
#         'entropy_bonus': 0.01,
#         'use_tvf': True,
#         'tvf_max_horizon': 1000,
#         'tvf_n_horizons': 64,
#         'tvf_coef': 0.01,
#         'vf_coef':0.5,              # this won't do anything (by default)
#         'tvf_gamma': 0.999,
#         'gamma': 0.999,
#         'policy_mini_batch_size': 2048,
#         'value_mini_batch_size': 256,
#     }
#
#     # this is very close to the old settings
#     # this worked **very** well, I should probably use this as the reference, and see if I can get TVF to perform
#     # as well as this. Maybe we can improve things a little by tweaking too?
#     add_job(
#         f"TVF_10_Regression",
#         run_name=f"ppo (alt)",
#         max_grad_norm=20.0,
#         agents=256,
#         n_steps=64,
#         use_tvf=False,
#         priority=150,
#         policy_mini_batch_size=1024,
#         value_mini_batch_size=1024,
#         value_epochs=4,
#         policy_epochs=4,
#         target_kl=1.0,
#         ppo_epsilon=0.1,
#         value_lr=2.5e-4,
#         policy_lr=2.5e-4,
#         gamma=0.999,
#         default_params=initial_args,
#     )
#
#     # this could just be a more realistic target to hit than 0.999?
#     # (actually 0.999 is probably fine)
#     add_job(
#         f"TVF_10_Regression",
#         run_name=f"ppo (alt_997)",
#         max_grad_norm=20.0,
#         agents=256,
#         n_steps=64,
#         use_tvf=False,
#         priority=150,
#         policy_mini_batch_size=1024,
#         value_mini_batch_size=1024,
#         value_epochs=4,
#         policy_epochs=4,
#         target_kl=1.0,
#         ppo_epsilon=0.1,
#         value_lr=2.5e-4,
#         policy_lr=2.5e-4,
#         gamma=0.997,
#         default_params=initial_args,
#     )
#
#     add_job(
#         # the is very close to the old settings
#         f"TVF_10_Regression",
#         run_name=f"tvf (alt)",
#         max_grad_norm=20.0,
#         agents=256,
#         n_steps=64,
#         use_tvf=True,
#         tvf_max_horizon=3000,
#         gamma=0.999,
#         tvf_gamma=0.999,
#         priority=150,
#         policy_mini_batch_size=1024,
#         value_mini_batch_size=1024,
#         value_epochs=4,
#         policy_epochs=4,
#         target_kl=1.0,
#         ppo_epsilon=0.1,
#         value_lr=2.5e-4,
#         policy_lr=2.5e-4,
#         default_params=initial_args,
#     )
#
#     add_job(
#         # the is very close to the old settings
#         f"TVF_10_Regression",
#         run_name=f"tvf (alt-16)",
#         max_grad_norm=20.0,
#         agents=256,
#         n_steps=64,
#         use_tvf=True,
#         tvf_lambda=-16, # try lower n-step.. this will be slow... but at least we wont be reusing the one bootstrap estimate...
#         tvf_max_horizon=3000,
#         gamma=0.999,
#         tvf_gamma=0.999,
#         priority=150,
#         policy_mini_batch_size=1024,
#         value_mini_batch_size=1024,
#         value_epochs=4,
#         policy_epochs=4,
#         target_kl=1.0,
#         ppo_epsilon=0.1,
#         value_lr=2.5e-4,
#         policy_lr=2.5e-4,
#         default_params=initial_args,
#     )
#
#     add_job(
#         # try make sure we learn value function fast enough...
#         f"TVF_10_Regression",
#         run_name=f"tvf (alt-16-hq)",
#         max_grad_norm=20.0,
#         agents=256,
#         n_steps=64,
#         use_tvf=True,
#         tvf_lambda=-16,
#         # try lower n-step.. this will be slow... but at least we wont be reusing the one bootstrap estimate...
#         tvf_max_horizon=3000,
#         gamma=0.999,
#         tvf_gamma=0.999,
#         priority=150,
#         policy_mini_batch_size=1024,
#         value_mini_batch_size=1024,
#         value_epochs=2,
#         policy_epochs=6,
#         target_kl=1.0,
#         ppo_epsilon=0.1,
#         value_lr=2.5e-4,
#         policy_lr=2.5e-4,
#         default_params=initial_args,
#     )
#
#     main_params = {
#         'checkpoint_every': int(5e6),
#         'workers': WORKERS,
#         'use_tvf': False,
#         'gamma': 0.999,
#         'env_name': 'DemonAttack',
#         'epochs': 50,       # because we filter out runs with low scores 50 epochs is fine
#         'vf_coef': 0.5,
#         'priority': -200,
#     }
#
#     search_params = {
#         'max_grad_norm': [0.5, 5, 20.0],    # should have no effect
#         'agents': [64, 128, 256],           # should have little effect
#         'n_steps': [16, 32, 64, 128],       # 16 was best from before, but unstable
#         'policy_mini_batch_size': [512, 1024, 2048],
#         'value_mini_batch_size': [512, 1024, 2048],
#         'value_epochs': [1, 2, 3, 4, 6, 8], # I have no idea about this one
#         'policy_epochs': [1, 2, 3, 4],      # I have no idea about this one
#         'target_kl': [1.0, 0.1, 0.01],      # 1.0 is effectively off
#         'ppo_epsilon': [0.05, 0.1, 0.2, 0.3], # I have only gotten <= 0.1 to work so far
#         'value_lr': [1e-4, 2.5e-4, 5e-4],   # any of these should work, but faster is better I guess?
#         'policy_lr': [1e-4, 2.5e-4, 5e-4],
#     }
#
#     # score threshold should be 200, but I want some early good results...
#     #random_search("TVF_10_Search_PPO", main_params, search_params, score_threshold=400, count=64)
#
#     # initial long horizon test... :)
#     tuned_args = {
#         'checkpoint_every': int(5e6),
#         'workers': WORKERS,
#         'env_name': 'DemonAttack',
#         'epochs': 50,
#         'max_grad_norm': 0.5,
#         'agents': 128,
#         'n_steps': 128,
#         'policy_mini_batch_size': 512,      # does this imply microbatching is broken?
#         'value_mini_batch_size': 512,
#         'value_epochs': 2,                  # this seems really low?
#         'policy_epochs': 4,
#         'target_kl': 0.01,
#         'ppo_epsilon': 0.05,
#         'value_lr': 5e-4,
#         'policy_lr': 2.5e-4,
#         'gamma': 0.999,
#     }
#
#     # initial long horizon test... :)
#     better_args = {
#         'checkpoint_every': int(5e6),
#         'workers': WORKERS,
#         'env_name': 'DemonAttack',
#         'epochs': 50,
#         'max_grad_norm': 5,             # more than before  [mode=5]
#         'agents': 128,                  #                   [mode=128]
#         'n_steps': 32,                  # less than before  [mode=128] (higher better)
#         'policy_mini_batch_size': 1024, # more than before  [mode=1024]
#         'value_mini_batch_size': 512,   #                   [mode=512] (lower better)
#         'value_epochs': 4,              # more than before  [mode=2]
#         'policy_epochs': 4,             #                   [mode=4] (higher better)
#         'target_kl': 0.10,              # much more than before (see if this is an issue...) [no mode]
#         'ppo_epsilon': 0.05,            # why so low?       [mode=0.05] (lower is better, but 0.3 works too?)
#         'value_lr': 1e-4,               # much lower        [no mode]
#         'policy_lr': 5e-4,              # higher? weird...  [mode=2.5e-4]
#         'gamma': 0.999,
#     }
#
#     # just a first test to get some early results
#
#     for gamma, tvf_max_horizon in zip(
#         #[0.99, 0.997, 0.999, 0.9997, 0.9999, 1.0],
#         #[300, 1000, 3000, 10000, 30000, 30000]
#         [0.9997, 0.9999, 1.0],
#         [10000, 30000, 30000]
#     ):
#         add_job(
#             f"TVF_10_LongHorizon",
#             run_name=f"ppo_gamma={gamma}",
#             use_tvf=False,
#             gamma=gamma,
#             default_params=tuned_args,
#             epochs=200,
#             priority=0,
#         )
#
#         add_job(
#             f"TVF_10_LongHorizon",
#             run_name=f"tvf_mc_gamma={gamma}",
#             use_tvf=True,
#             tvf_hidden_units=128,                   # mostly a performance optimization
#             tvf_n_horizons=128,
#             tvf_lambda=1.0,
#             tvf_coef=0.01,
#             tvf_max_horizon=tvf_max_horizon,
#             gamma=gamma,
#             tvf_gamma=gamma,
#             default_params=tuned_args,
#             epochs=200,
#             priority=0,
#         )
#
#         add_job(
#             f"TVF_10_LongHorizon",
#             run_name=f"tvf_mc_better_gamma={gamma}",
#
#             use_tvf=True,
#             tvf_hidden_units=128,  # mostly a performance optimization
#             tvf_n_horizons=128,
#
#             tvf_loss_weighting="advanced",
#             tvf_h_scale="squared",
#
#             tvf_lambda=1.0,
#             tvf_coef=0.01,
#             tvf_max_horizon=tvf_max_horizon,
#             gamma=gamma,
#             tvf_gamma=gamma,
#             default_params=better_args,
#             epochs=200,
#             priority=0,
#         )
#
#         add_job(
#             # note: would be interesting to try setting tvf_gamma to 1.
#             f"TVF_10_LongHorizon",
#             run_name=f"tvf_16_gamma={gamma}",
#             use_tvf=True,
#             tvf_hidden_units=128,                   # mostly a performance optimization
#             tvf_n_horizons=128,
#             tvf_lambda=-16,
#             tvf_max_horizon=tvf_max_horizon,
#             tvf_coef=0.01,
#             gamma=gamma,
#             tvf_gamma=gamma,
#             default_params=tuned_args,
#             priority=0,  # turn off for the moment...
#         )
#
#
#     # bundles...
#     add_job(
#         f"TVF_10_Eval",
#         run_name=f"bundle_0",
#         use_tvf=False,
#         epochs=200,
#         default_params=tuned_args,
#         priority=200,
#     )
#     add_job(
#         f"TVF_10_Eval",
#         run_name=f"bundle_1",
#         use_tvf=False,
#         epochs=200,
#         default_params=better_args,
#         priority=200,
#     )
#
#     #slow and stead wins the race
#     add_job(
#         f"TVF_10_Eval",
#         run_name=f"bundle_2",
#         use_tvf=False,
#         epochs=200,
#
#         # mostly set from modes
#         max_grad_norm=5,
#         agents=128,
#         n_steps=128,
#         policy_mini_batch_size=1024,
#         value_mini_batch_size=512,
#         value_epochs=2,
#         policy_epochs=4,
#         target_kl=0.01,
#         ppo_epsilon=0.05,
#         value_lr=1e-4,
#         policy_lr=2.5e-4,
#         gamma=0.999,
#
#         default_params=better_args,
#         priority=200,
#     )
#
#     # these had tvf_coef set wrong... (it was 0.1, should be 0.01)
#     add_job(
#         f"TVF_10_Eval",
#         run_name=f"tvf_1_adv",
#
#         use_tvf=True,
#         tvf_max_horizon=3000,
#         tvf_hidden_units=128,
#         tvf_n_horizons=16,
#
#         tvf_loss_weighting="advanced",
#         tvf_h_scale="squared",
#
#         epochs=200,
#         default_params=better_args,
#         priority=100,
#     )
#     #
#     # add_job(
#     #     f"TVF_10_Eval",
#     #     run_name=f"tvf_1_std",
#     #
#     #     use_tvf=True,
#     #     tvf_max_horizon=3000,
#     #     tvf_hidden_units=128,
#     #     tvf_n_horizons=16,
#     #
#     #     epochs=200,
#     #     default_params=better_args,
#     #     priority=100,
#     # )
#
#
#     add_job(
#         f"TVF_10_Eval",
#         run_name=f"tvf_1_adv_coef",
#
#         use_tvf=True,
#         tvf_max_horizon=3000,
#         tvf_hidden_units=128,
#         tvf_n_horizons=16,
#         tvf_coef=0.01,
#
#         tvf_loss_weighting="advanced",
#         tvf_h_scale="squared",
#
#         epochs=200,
#         default_params=better_args,
#         priority=100,  # turn off for the moment...
#     )
#
#     add_job(
#         f"TVF_10_Eval",
#         run_name=f"tvf_1_std_coef",
#
#         use_tvf=True,
#         tvf_max_horizon=3000,
#         tvf_hidden_units=128,
#         tvf_n_horizons=16,
#         tvf_coef=0.01,
#
#         epochs=200,
#         default_params=better_args,
#         priority=100,  # turn off for the moment...
#     )
#
# def setup_experiments_10_eval():
#     # initial long horizon test... :)
#     tuned_args = {
#         'checkpoint_every': int(5e6),
#         'workers': WORKERS,
#         'env_name': 'DemonAttack',
#         'epochs': 50,
#         'max_grad_norm': 0.5,
#         'agents': 128,
#         'n_steps': 128,
#         'policy_mini_batch_size': 512,  # does this imply microbatching is broken?
#         'value_mini_batch_size': 512,
#         'value_epochs': 2,  # this seems really low?
#         'policy_epochs': 4,
#         'target_kl': 0.01,
#         'ppo_epsilon': 0.05,
#         'value_lr': 5e-4,
#         'policy_lr': 2.5e-4,
#         'gamma': 0.999,
#     }
#
#     # initial long horizon test... :)
#     better_args = {
#         'checkpoint_every': int(5e6),
#         'workers': WORKERS,
#         'env_name': 'DemonAttack',
#         'epochs': 50,
#         'max_grad_norm': 5,  # more than before  [mode=5]
#         'agents': 128,  # [mode=128]
#         'n_steps': 32,  # less than before  [mode=128] (higher better)
#         'policy_mini_batch_size': 1024,  # more than before  [mode=1024]
#         'value_mini_batch_size': 512,  # [mode=512] (lower better)
#         'value_epochs': 4,  # more than before  [mode=2]
#         'policy_epochs': 4,  # [mode=4] (higher better)
#         'target_kl': 0.10,  # much more than before (see if this is an issue...) [no mode]
#         'ppo_epsilon': 0.05,  # why so low?       [mode=0.05] (lower is better, but 0.3 works too?)
#         'value_lr': 1e-4,  # much lower        [no mode]
#         'policy_lr': 5e-4,  # higher? weird...  [mode=2.5e-4]
#         'gamma': 0.999,
#     }
#
#     for env_name in ["Alien", "BankHeist", "CrazyClimber"]:
#         #this was bundle_0 which performed quite well in my initial tests...
#         #wait for jobs to finish and change the folders they run in...
#         add_job(
#             f"TVF_10_Test_Bundle_0",
#             run_name=f"{env_name}",
#             env_name=env_name,
#             use_tvf=False,
#             epochs=50,
#             default_params=tuned_args,
#             priority=50,
#         )
#
#
#     # next tests:
#     # wait for TVF runs to finish, pick best one then try long horizon test with PPO and TVF
#
#     for env_name in ["Alien", "BankHeist", "CrazyClimber", "DemonAttack"]:
#         #this was bundle_0 which performed quite well in my initial tests...
#         #wait for jobs to finish and change the folders they run in...
#         add_job(
#             f"TVF_10_Test_TVF_tuned",
#             run_name=f"{env_name}",
#             env_name=env_name,
#             epochs=50,
#
#             use_tvf=True,
#             tvf_hidden_units=128,  # mostly a performance optimization
#             tvf_n_horizons=128,
#             tvf_lambda=1.0,
#             tvf_coef=0.01,
#             tvf_max_horizon=3000,
#             gamma=0.999,
#             tvf_gamma=0.999,
#
#             default_params=tuned_args,
#             priority=50,
#         )
#
#         add_job(
#             f"TVF_10_Test_TVF_tuned_adv",
#             run_name=f"{env_name}",
#             env_name=env_name,
#             epochs=50,
#
#             use_tvf=True,
#             tvf_hidden_units=128,  # mostly a performance optimization
#             tvf_n_horizons=128,
#             tvf_lambda=1.0,
#             tvf_coef=0.01,
#             tvf_max_horizon=3000,
#             gamma=0.999,
#             tvf_gamma=0.999,
#
#             tvf_loss_weighting="advanced",
#             tvf_h_scale="squared",
#
#             default_params=tuned_args,
#             priority=-50,
#         )
#
#         add_job(
#             f"TVF_10_Test_TVF_better",
#             run_name=f"{env_name}",
#             env_name=env_name,
#             epochs=50,
#
#             use_tvf=True,
#             tvf_hidden_units=128,  # mostly a performance optimization
#             tvf_n_horizons=128,
#             tvf_lambda=1.0,
#             tvf_coef=0.01,
#             tvf_max_horizon=3000,
#             gamma=0.999,
#             tvf_gamma=0.999,
#
#             default_params=better_args,
#             priority=-50,
#         )
#
#
#         add_job(
#             f"TVF_10_Test_TVF_better_adv",
#             run_name=f"{env_name}",
#             env_name=env_name,
#             epochs=50,
#
#             use_tvf=True,
#             tvf_hidden_units=128,  # mostly a performance optimization
#             tvf_n_horizons=128,
#             tvf_lambda=1.0,
#             tvf_coef=0.01,
#             tvf_max_horizon=3000,
#             gamma=0.999,
#             tvf_gamma=0.999,
#
#             tvf_loss_weighting="advanced",
#             tvf_h_scale="squared",
#
#             default_params=better_args,
#             priority=-50,
#         )
#

def setup_experiments_11():


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
        'value_lr': 5e-4,
        'policy_lr': 2.5e-4,
        'gamma': 0.999,
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

    add_job(
        f"TVF_11_Regression",
        run_name=f"ppo (tuned)",
        default_params=tuned_args,
    )

    add_job(
        f"TVF_11_Regression",
        run_name=f"ppo (better)",
        default_params=better_args,
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
    )

def setup_experiments_11_eval():
    # initial long horizon test... :)
    tuned_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'env_name': 'DemonAttack',
        'epochs': 50,
        'max_grad_norm': 0.5,
        'agents': 128,
        'n_steps': 128,
        'policy_mini_batch_size': 512,  # does this imply microbatching is broken?
        'value_mini_batch_size': 512,
        'value_epochs': 2,  # this seems really low?
        'policy_epochs': 4,
        'target_kl': 0.01,
        'ppo_epsilon': 0.05,
        'value_lr': 5e-4,
        'policy_lr': 2.5e-4,
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

    # initial long horizon test... :)
    better_args = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'env_name': 'DemonAttack',
        'epochs': 50,
        'max_grad_norm': 5,  # more than before  [mode=5]
        'agents': 128,  # [mode=128]
        'n_steps': 32,  # less than before  [mode=128] (higher better)
        'policy_mini_batch_size': 1024,  # more than before  [mode=1024]
        'value_mini_batch_size': 512,  # [mode=512] (lower better)
        'value_epochs': 4,  # more than before  [mode=2]
        'policy_epochs': 4,  # [mode=4] (higher better)
        'target_kl': 0.10,  # much more than before (see if this is an issue...) [no mode]
        'ppo_epsilon': 0.05,  # why so low?       [mode=0.05] (lower is better, but 0.3 works too?)
        'value_lr': 1e-4,  # much lower        [no mode]
        'policy_lr': 5e-4,  # higher? weird...  [mode=2.5e-4]
        'gamma': 0.999,
    }

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
        f"TVF_11_Periodic",
        run_name=f"tvf_tuned_adv",
        epochs=50,

        default_params=tvf_tuned_adv_args,
        priority=200,
    )

    for chunk_size in [5,10,20,50]:
        add_job(
            f"TVF_11_Chunking",
            run_name=f"chunk={chunk_size}",
            epochs=50,
            chunk_size=chunk_size,

            default_params=tvf_tuned_adv_args,
            priority=200,
        )



    for tvf_n_horizons in [16, 32, 64, 128]:
        add_job(
            f"TVF_11_n_horizons",
            run_name=f"tvf_n_horizons={tvf_n_horizons}",
            epochs=50,
            tvf_n_horizons=tvf_n_horizons,
            default_params=tvf_tuned_adv_args,
            priority=20,
        )


def random_search_11_ppo():
    main_params = {
        'checkpoint_every': int(5e6),
        'workers': WORKERS,
        'use_tvf': False,
        'gamma': 0.999,
        'env_name': 'DemonAttack',
        'epochs': 50,       # because we filter out runs with low scores 50 epochs is fine
        'vf_coef': 0.5,
        'priority': -200,
    }

    search_params = {
        'max_grad_norm': [0.5, 5, 20.0],    # should have no effect
        'agents': [64, 128, 256],           # should have little effect
        'n_steps': [16, 32, 64, 128],       # 16 was best from before, but unstable
        'policy_mini_batch_size': [512, 1024, 2048],
        'value_mini_batch_size': [512, 1024, 2048],
        'value_epochs': [1, 2, 3, 4, 6, 8],
        'policy_epochs': [1, 2, 3, 4],
        'target_kl': [1.0, 0.1, 0.01],      # 1.0 is effectively off
        'ppo_epsilon': [0.05, 0.1, 0.2, 0.3], # I have only gotten <= 0.1 to work so far
        'value_lr': [1e-4, 2.5e-4, 5e-4],   # any of these should work, but faster is better I guess?
        'policy_lr': [1e-4, 2.5e-4, 5e-4],
    }

    # score threshold should be 200, but I want some early good results...
    random_search(
        "TVF_11_Search_PPO",
        main_params,
        search_params,
        count=64,
        envs=['BattleZone', 'DemonAttack', 'Amidar']
    )



if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    id = 0
    job_list = []
    setup_experiments_11()
    setup_experiments_11_eval()
    #random_search_11_ppo()

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