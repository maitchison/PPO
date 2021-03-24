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

        # stub: work on partial ones first...
        #priority = priority - self.get_completed_epochs()
        priority = priority + self.get_completed_epochs()

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
        'workers': 8,
        'tvf_gamma': 0.997,
        'gamma': 0.997,
        'n_mini_batches': 32,
    }

    for tvf_max_horizon in [300, 1000, 3000]:
        add_job(
            "TVF_6A",
            run_name=f"tvf_mh={tvf_max_horizon}",
            tvf_max_horizon=tvf_max_horizon,
            priority=50,
            **default_args
        )

    for agents in [128, 256, 512]:
        add_job(
            "TVF_6B",
            run_name=f"agents={agents}",
            tvf_max_horizon=1000,
            agents=agents,
            **{k:v for k,v in default_args.items() if k != "agents"}
        )

    for n_steps in [64, 128, 256]:
        add_job(
            "TVF_6C",
            run_name=f"n_steps={n_steps}",
            tvf_max_horizon=1000,
            n_steps=n_steps,
            **{k:v for k,v in default_args.items() if k != "n_steps"}
        )

    for distribution in ["uniform", "linear", "first_and_last"]:
        add_job(
            "TVF_6D",
            run_name=f"tvf_sample_dist={distribution} samples=32",
            tvf_max_horizon=1000,
            tvf_sample_dist=distribution,
            tvf_n_horizons=32,
            **{k:v for k,v in default_args.items() if k != "tvf_n_horizons"}
        )


def setup_experiments5():

    # 5A Just to make sure we didn't break anything when we changed how sampling works etc
    # I did change gamma to be the same as tvf_gamma which enables the MC optimization and also
    # might improve training, as otherwise noise at distant horizons gets scaled up a lot

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
        'workers': 8,
        'tvf_gamma': 0.997,
        'gamma': 0.997,
        'n_mini_batches': 32, # should be 8 I think...
    }

    for tvf_max_horizon in [1000, 2000, 4000]:
        add_job(
            "TVF_5A",
            run_name=f"tvf_mh={tvf_max_horizon}",
            tvf_max_horizon=tvf_max_horizon,
            priority=30,
            **default_args
        )

    for tvf_max_horizon in [1000, 2000, 4000]:
        add_job(
            "TVF_5A",
            run_name=f"tvf_mh={tvf_max_horizon} (MSE)",
            tvf_max_horizon=tvf_max_horizon,
            tvf_loss_func="mse",
            priority=50,
            **default_args
        )

    # check n_step
    for tvf_n_step in [0, 1, 2, 4, 8, 16, 32, 64, 128]:
        add_job(
            "TVF_5B",
            run_name=f"tvf_n_step={tvf_n_step}",
            tvf_max_horizon=1000,
            tvf_lambda=-tvf_n_step,
            priority=30,
            **default_args
        )

    # check lambda style
    # (algorithm was broken)
    # for tvf_lambda in [0.9, 0.95, 0.99]:
    #     add_job(
    #         "TVF_5C",
    #         run_name=f"tvf_lambda={tvf_lambda}",
    #         tvf_max_horizon=1000,
    #         tvf_lambda=tvf_lambda,
    #         priority=30,
    #         **default_args
    #     )

    # check effect of sampling
    for tvf_n_horizons in [16, 32, 64, 128, 256, 512, 1000]:
        add_job(
            "TVF_5D",
            run_name=f"tvf_n_horizons={tvf_n_horizons}",
            tvf_max_horizon=1000,
            tvf_n_horizons=tvf_n_horizons,
            max_micro_batch_size=256 if tvf_n_horizons==1000 else 512,
            priority=50,
            **{k:v for k,v in default_args.items() if k != "tvf_n_horizons"}
        )

    # check effect of sampling
    for tvf_n_horizons in [16, 32, 64, 128, 256, 512, 1000]:
        add_job(
            "TVF_5D",
            run_name=f"tvf_n_horizons={tvf_n_horizons} dist=linear",
            tvf_max_horizon=1000,
            tvf_n_horizons=tvf_n_horizons,
            tvf_sample_dist='linear',
            max_micro_batch_size=256 if tvf_n_horizons == 1000 else 512,
            priority=50,
            **{k: v for k, v in default_args.items() if k != "tvf_n_horizons"}
        )


    # for tvf_max_horizon in [1000, 2000, 4000]:
    #     add_job(
    #         "TVF_5A",
    #         run_name=f"tvf_mh={tvf_max_horizon} mb=8",
    #         tvf_max_horizon=tvf_max_horizon,
    #         priority=0,
    #         n_mini_batches=8,
    #         **{k:v for k,v in default_args.items() if k != "n_mini_batches"}
    #     )
    #
    #
    #     # 5B All kinds of tricks... horizon warmup
    # for hw in [0.1, 0.5]:
    #     add_job(
    #         "TVF_5B",
    #         run_name=f"tvf_mh={4000} hw={hw}",
    #         tvf_max_horizon=4000,
    #         tvf_horizon_warmup=hw,
    #         **default_args
    #     )
    #
    # # 5B All kinds of tricks... loss
    # for loss_fn in ["huber", "mse"]:
    #     add_job(
    #         "TVF_5B",
    #         run_name=f"tvf_mh={4000} lf={loss_fn}",
    #         tvf_max_horizon=4000,
    #         tvf_loss_func=loss_fn,
    #         **default_args
    #     )
    #
    # # 5B All kinds of tricks... dist
    # for sd in ["linear"]:
    #     add_job(
    #         "TVF_5B",
    #         run_name=f"tvf_mh={4000} sd={sd}",
    #         tvf_max_horizon=4000,
    #         tvf_sample_dist=sd,
    #         **default_args
    #     )
    #
    # # 5C Just checking microbatching works
    # for mmbs in [32, 128, 512, 2048]:  # 1024 will not micro_batch training, and 2048 will not micro_batch the final value calculation
    #     add_job(
    #         "TVF_5C",
    #         run_name=f"tvf_mh={4000} mmmbs={mmbs}",
    #         tvf_max_horizon=4000,
    #         n_mini_batches=32,
    #         max_micro_batch_size=mmbs,
    #         **{k: v for k, v in default_args.items() if k != "n_mini_batches"},
    #     )

    # 5B All kinds of tricks... minibatches
    # for mb in [4, 8, 16]:
    #     add_job(
    #         "TVF_5B",
    #         run_name=f"tvf_mh={4000} mb={sd}",
    #         tvf_max_horizon=4000,
    #         n_mini_batches=mb,
    #         **default_args
    #     )

    # 5B All kinds of tricks... agents
    # for agents in [128, 256, 512]:
    #     add_job(
    #         "TVF_5B",
    #         run_name=f"tvf_mh={4000} agents={agents}",
    #         tvf_max_horizon=4000,
    #         agents=agents,
    #         **default_args
    #     )


def random_search(run, main_params, search_params, count=128):

    for i in range(count):
        params = {}
        np.random.seed(i)
        for k, v in search_params.items():
            params[k] = np.random.choice(v)

        add_job(run, run_name=f"{i:04d}", chunked=False, **main_params, **params)


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
        'epochs': 25,
        'use_tvf': True,
        'tvf_advantage': True,
        'vf_coef': 0.0,
        'tvf_epsilon': 0.1,
        'workers': 8,
        'time_aware': False,
    }

    search_params = {
        'agents': [64, 128, 256],
        'n_steps': [32, 64, 128, 256, 512],
        'max_grad_norm': [0.5, 5.0, None],
        'entropy_bonus': [0.3, 0.01, 0.003],
        'tvf_coef': [1.0, 0.3, 0.1, 0.03, 0.01],
        'tvf_n_horizons': [30, 100, 250],
        'tvf_max_horizon': [300, 1000, 3000],
        'gamma': [0.99, 0.997, 0.999],
        'n_mini_batches': [8, 16, 32, 64],
        'adam_epsilon': [1e-5, 1e-8],
        'learning_rate': [1e-3, 2.5e-4, 1e-4],
        #'tvf_lambda': [0, 1, 0.9, 0.95, 0.99, -4, -8],
        'tvf_lambda': [1.0],  # 1 is faster and more likely to work, and we can experiment witt the others later (they don't seem to work atm)
        'tvf_loss_func': ['nlp', 'huber', 'mse'],
        'tvf_sample_dist': ['linear', 'uniform'],
        'tvf_horizon_warmup': [0, 0.1, 0.5],
        'tvf_horizon_transform': [None, "log"],
    }

    random_search("tvf_search", main_params, search_params)


if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    id = 0
    job_list = []
    #setup_mvh4()
    setup_experiments5()
    setup_experiments6()
    #setup_tvf_random_search()

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