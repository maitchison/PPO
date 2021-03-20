import numpy
import os
import sys
import pandas as pd
import json
import time

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

    if 'ignore_device' not in kwargs:
        if HOST_NAME != 'matthew-desktop':
            kwargs['ignore_device'] = "[0, 1]"

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

        params_part = " ".join([f"--{k}={nice_format(v)}" for k, v in self.params.items() if k not in ["env_name"]])
        params_part_lined = "\n".join([f"--{k}={nice_format(v)}" for k, v in self.params.items() if k not in ["env_name"]])

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

def setup_mvh():

    # just a quick test to see how truncated value does for actual policy learning
    # note: we learn 0.99 at 300 timesteps, but then rediscount these to see what happens.
    for env in ["Breakout"]:
        for gamma in [0.99, 0.997, 0.999]:
            add_job(
                "TVF_3A".format(env),
                run_name=f"gamma={gamma}",
                env_name=env,
                checkpoint_every=int(5e6),
                epochs=50,
                agents=256,

                use_tvf=True,
                tvf_coef=0.1,
                tvf_max_horizon=300,
                tvf_n_horizons=100,
                tvf_gamma=0.99,
                tvf_advantage=True,

                workers=8,
                gamma=gamma,
                time_aware=False,
                priority=20,
            )

    # just a quick test to see how truncated value does for actual policy learning
    # note: we learn 0.997 at 500 timesteps, but then rediscount these to see what happens.
    for env in ["Breakout"]:
        for gamma in [0.99, 0.997, 0.999]:
            add_job(
                "TVF_3B".format(env),
                run_name=f"gamma={gamma}",
                env_name=env,
                checkpoint_every=int(5e6),
                epochs=50,
                agents=256,

                use_tvf=True,
                tvf_coef=0.1,
                tvf_max_horizon=500,
                tvf_n_horizons=100,
                tvf_gamma=0.997,
                tvf_advantage=True,

                workers=8,
                gamma=gamma,
                time_aware=False,
                priority=20,
            )

    # hyperparameter search for short horizon...
    # this is almost 250 experiments... this will take a long time...
    # at 20 hours per we have 500 hours using 8 workers is 3 days... might be ok.. maybe I should do axis search instead...
    # actually just wait the 3 days it'll be fine.
    def make_TVF_3B_job(run_name, env="Breakout", gamma=0.99, n_steps=128, tvf_coef=0.01, tvf_epsilon=0.01, tvf_max_horizon=300, tvf_n_horizons=30):
        add_job(
            "TVF_3_HPS",
            run_name=run_name,
            env_name=env,
            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,

            n_steps=n_steps,
            gamma=gamma,

            use_tvf=True,
            tvf_coef=tvf_coef,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=tvf_n_horizons,
            tvf_epsilon=tvf_epsilon,
            tvf_gamma=0.99,
            tvf_advantage=True,

            workers=8,
            time_aware=False,
            priority=5,
        )

    # for n_steps in [64, 128, 256]:
    #     make_TVF_3B_job(f"n_steps={n_steps}", n_steps=n_steps)
    # for tvf_coef in [0.1, 0.01, 0.001]:
    #     make_TVF_3B_job(f"tvf_coef={tvf_coef}", tvf_coef=tvf_coef)
    # for tvf_epsilon in [0.1, 0.01, 0.001]:
    #     make_TVF_3B_job(f"tvf_epsilon={tvf_epsilon}", tvf_epsilon=tvf_epsilon)
    # for tvf_max_horizon in [100, 300, 1000]:
    #     make_TVF_3B_job(f"tvf_max_horizon={tvf_max_horizon}", tvf_max_horizon=tvf_max_horizon)
    # for tvf_n_horizons in [10, 30, 100]:
    #     make_TVF_3B_job(f"tvf_n_horizons={tvf_n_horizons}", tvf_n_horizons=tvf_n_horizons)

    # switched to a TD(\lambda) style estimates, might be less noisy...
    # I test here short/med/long acting horizon with short/long learning horizon

    for env in ["Breakout"]:
        for gamma in [0.99, 0.997, 0.999]:
            for tvf_gamma in [0.99, 0.999]:
                add_job(
                    "TVF_3C".format(env),
                    run_name=f"gamma={gamma} tvf_gamma={tvf_gamma}",
                    env_name=env,
                    checkpoint_every=int(5e6),
                    epochs=50,
                    agents=256,

                    use_tvf=True,
                    tvf_coef=0.1,
                    tvf_max_horizon=100 if tvf_gamma == 0.99 else 1000,
                    tvf_n_horizons=100,
                    tvf_gamma=tvf_gamma,
                    tvf_advantage=True,

                    workers=8,
                    gamma=gamma,
                    time_aware=False,
                    priority=20,
                )

    # using td(0) and no dist
    for env in ["Breakout"]:
        for gamma in [0.99, 0.999]:
            for tvf_gamma in [0.99, 0.999]:
                add_job(
                    "TVF_3D".format(env),
                    run_name=f"gamma={gamma} tvf_gamma={tvf_gamma}",
                    env_name=env,
                    checkpoint_every=int(5e6),
                    epochs=50,
                    agents=256,

                    use_tvf=True,
                    tvf_coef=0.1,
                    tvf_max_horizon=100 if tvf_gamma == 0.99 else 500,
                    tvf_n_horizons=100,
                    tvf_gamma=tvf_gamma,
                    tvf_advantage=True,
                    tvf_distributional=False,
                    tvf_lambda=0,

                    workers=8,
                    gamma=gamma,
                    time_aware=False,
                    priority=25,
                )

    # going back to our MC update
    for env in ["Breakout"]:
        for coef in [1, 0.1, 0.01]:
            add_job(
                "TVF_3E".format(env),
                run_name=f"coef={coef}",
                env_name=env,
                checkpoint_every=int(5e6),
                epochs=50,
                agents=256,

                use_tvf=True,
                tvf_coef=coef,
                tvf_max_horizon=300,
                tvf_n_horizons=100,
                tvf_gamma=0.99,
                tvf_advantage=True,

                workers=8,
                gamma=0.99,
                time_aware=False,
                priority=20,
            )

            add_job(
                "TVF_3E".format(env),
                run_name=f"coef={coef} vf_coef=0",
                env_name=env,
                checkpoint_every=int(5e6),
                epochs=50,
                agents=256,

                use_tvf=True,
                tvf_coef=coef,
                tvf_max_horizon=300,
                tvf_n_horizons=100,
                tvf_gamma=0.99,
                tvf_advantage=True,
                vf_coef=0,

                workers=8,
                gamma=0.99,
                time_aware=False,
                priority=20,
            )

    # going back to our MC update, with distributional back in
    # 3F turned into a bit a of a hp search. this is tvf_coef
    for env in ["Breakout"]:
        for tvf_coef in [1, 0.1, 0.03, 0.01]:
            add_job(
                "TVF_3F".format(env), # named incorrectly...
                run_name=f"coef={tvf_coef}",
                env_name=env,
                checkpoint_every=int(5e6),
                epochs=50,
                agents=256,

                use_tvf=True,
                tvf_coef=tvf_coef,
                tvf_max_horizon=300,
                tvf_n_horizons=100,
                tvf_gamma=0.99,
                tvf_advantage=True,
                tvf_distributional=True,
                vf_coef=0,

                workers=8,
                gamma=0.99,
                time_aware=False,
                priority=25,
            )

    # going back to our MC update, with distributional back in
    # 3F turned into a bit a of a hp search. this is tvf_coef
    for env in ["Breakout"]:
        for tvf_coef in [1, 0.1, 0.03, 0.01]:
            add_job(
                "TVF_3F1".format(env),
                run_name=f"tvf_coef={tvf_coef}",
                env_name=env,
                checkpoint_every=int(5e6),
                epochs=50,
                agents=256,

                use_tvf=True,
                tvf_coef=tvf_coef,
                tvf_max_horizon=300,
                tvf_n_horizons=100,
                tvf_gamma=0.99,
                tvf_advantage=True,
                tvf_distributional=True,
                vf_coef=0,

                workers=8,
                gamma=0.99,
                time_aware=False,
                priority=15,
            )

    # going back to our MC update, with distributional back in
    # 3F turned into a bit a of a hp search. this is value function learning
    env = "Breakout"
    for vf_coef in [1.0, 0.5, 0]:
        add_job(
            "TVF_3F2".format(env),
            run_name=f"vf_coef={vf_coef}",
            env_name=env,
            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=300,
            tvf_n_horizons=100,
            tvf_gamma=0.99,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=vf_coef,

            workers=8,
            gamma=0.99,
            time_aware=False,
            priority=15,
        )

    # going back to our MC update, with distributional back in
    # 3F turned into a bit a of a hp search. this is distributional epsilon
    env = "Breakout"
    tvf_coef = 0.03
    for tvf_epsilon in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
            add_job(
                "TVF_3F3".format(env),
                run_name=f"tvf_epsilon={tvf_epsilon}",
                env_name=env,
                checkpoint_every=int(5e6),
                epochs=50,
                agents=256,

                use_tvf=True,
                tvf_coef=tvf_coef,
                tvf_max_horizon=300,
                tvf_n_horizons=100,
                tvf_gamma=0.99,
                tvf_advantage=True,
                tvf_distributional=True,
                tvf_epsilon=tvf_epsilon,

                workers=8,
                gamma=0.99,
                time_aware=False,
                priority=15,
            )

    # going back to our MC update, with distributional back in
    # this time we learn the ext_value as well
    # higher epsilon to stablize training at end
    env = "Breakout"
    tvf_coef = 0.03
    for vf_coef in [0, 0.5]:
        for gamma in [1, 0.999, 0.997, 0.99]:
            add_job(
                "TVF_3H".format(env),
                run_name=f"tvf_coef={tvf_coef} vf_coef={vf_coef} gamma={gamma}",
                env_name=env,
                checkpoint_every=int(5e6),
                epochs=50,
                agents=256,

                use_tvf=True,
                tvf_coef=tvf_coef,
                tvf_max_horizon=300,
                tvf_n_horizons=100,
                tvf_gamma=1.0,
                tvf_advantage=True,
                tvf_distributional=True,
                vf_coef=vf_coef,
                tvf_epsilon=0.1,

                workers=8,
                gamma=gamma,
                time_aware=False,
                priority=20,
            )


def setup_mvh4():

    # found a big in MC estimates, so going to try again clean
    env = "Breakout"

    # goals
    # verify MC works
    # find tvf_coef
    # find tvf_episilon

    # extra...
    # check vf_coef matters
    # try learning non-discounted returns
    # try learning non-discounted returns and rediscounting

    # first experiment just see how the new MC code is doing and look for a good tvf_coef
    for tvf_coef in [1, 0.3, 0.1, 0.03, 0.01]:
        add_job(
            "TVF_4A".format(env),
            run_name=f"tvf_coef={tvf_coef}",
            env_name=env,
            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,

            use_tvf=True,
            tvf_coef=tvf_coef,
            tvf_max_horizon=300,
            tvf_n_horizons=100,
            tvf_gamma=0.99,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=0.99,
            time_aware=False,
            priority=30,
        )

    for tvf_coef in [1, 0.3, 0.1, 0.03, 0.01]:
        add_job(
            "TVF_4B".format(env),
            run_name=f"tvf_coef={tvf_coef}",
            env_name=env,
            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,

            use_tvf=True,
            tvf_coef=tvf_coef,
            tvf_max_horizon=300,
            tvf_n_horizons=100,
            tvf_gamma=0.99,
            tvf_advantage=True,
            tvf_distributional=False,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=0.99,
            time_aware=False,
            priority=30,
        )

    for tvf_epsilon in [1.0, 0.3, 0.1, 0.03, 0.01]:
        add_job(
            "TVF_4C".format(env),
            run_name=f"tvf_epsilon={tvf_epsilon}",
            env_name=env,
            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=300,
            tvf_n_horizons=100,
            tvf_gamma=0.99,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=tvf_epsilon,

            workers=8,
            gamma=0.99,
            time_aware=False,
            priority=20,
        )

    # 4D can we learn long undiscounted horizons then rediscount?
    tvf_max_horizon = 1000
    tvf_gamma = 1.0
    for gamma in [0.99, 0.997, 0.999]:
        add_job(
            "TVF_4D".format(env),
            run_name=f"tvf_mh={tvf_max_horizon} tvf_gamma={tvf_gamma} gamma={gamma}",
            env_name=env,
            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=100,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=10,
        )

    # 4D can we learn short undiscounted horizons then rediscount?
    tvf_max_horizon = 300
    tvf_gamma = 1.0
    for gamma in [0.99, 0.997, 0.999]:
        add_job(
            "TVF_4E".format(env),
            run_name=f"tvf_mh={tvf_max_horizon} tvf_gamma={tvf_gamma} gamma={gamma}",
            env_name=env,
            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=100,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=5,
        )

    # 4F another attempt at long horizon
    tvf_gamma = 1.0
    gamma = 0.997
    for tvf_max_horizon in [100, 300, 500]: # 1000 is too much without sampling...
        add_job(
            "TVF_4F".format(env),
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name=env,
            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,

            max_grad_norm = 5.0,
            entropy_bonus = 0.03,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=tvf_max_horizon,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=5,
        )

    # 4G another GAE estimator...
    tvf_gamma = 0.997
    gamma = 0.997 # see if this helps a bit...)
    for tvf_max_horizon in [100, 300, 500]:
        add_job(
            "TVF_4G".format(env),
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name=env,
            checkpoint_every=int(5e6),
            epochs=50,
            agents=256,

            max_grad_norm=5.0,
            entropy_bonus=0.03,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=tvf_max_horizon,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_real_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=25,
        )

    # 4H another attempt at long horizon
    tvf_gamma = 1.0
    gamma = 0.997
    for tvf_max_horizon in [100, 300, 500, 1000, 2000]:
        add_job(
            "TVF_4H".format(env),
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name=env,
            checkpoint_every=int(5e6),
            epochs=50,
            agents=128,
            n_steps=256,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=min(tvf_max_horizon, 250),
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=10,
        )

    # 4H another attempt at long horizon
    tvf_gamma = 0.997 # this helps ensure that shorter horizons matter more in terms of loss.
    gamma = 0.999
    for tvf_max_horizon in [500, 1000, 2000, 4000]:
        add_job(
            "TVF_4I".format(env),
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name=env,
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=min(tvf_max_horizon, 250),
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=10,
        )

    # 4J alien
    tvf_gamma = 0.997  # this helps ensure that shorter horizons matter more in terms of loss.
    gamma = 0.999
    for tvf_max_horizon in [500, 1000, 2000, 4000]:
        add_job(
            "TVF_4J",
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name="Alien",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=min(tvf_max_horizon, 250),
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=45 if tvf_max_horizon <= 2000 else 0,
        )
    for gamma in [0.99, 0.997, 0.999]:
        add_job(
            "TVF_4J",
            run_name=f"baseline gamma={gamma}",
            env_name="Alien",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=False,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=15,
        )

    # 4K montezuma
    tvf_gamma = 0.997  # this helps ensure that shorter horizons matter more in terms of loss.
    gamma = 0.999
    for tvf_max_horizon in [500, 1000, 2000, 4000]:
        add_job(
            "TVF_4K",
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name="MontezumaRevenge",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=min(tvf_max_horizon, 250),
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=45 if tvf_max_horizon <= 2000 else 0,
        )
    for gamma in [0.99, 0.997, 0.999]:
        add_job(
            "TVF_4K",
            run_name=f"baseline gamma={gamma}",
            env_name="MontezumaRevenge",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=False,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=15,
        )


    # 4L seaquest
    tvf_gamma = 0.997  # this helps ensure that shorter horizons matter more in terms of loss.
    gamma = 0.999
    for tvf_max_horizon in [500, 1000, 2000, 4000]:
        add_job(
            "TVF_4L",
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name="Seaquest",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=min(tvf_max_horizon, 250),
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=0,
        )
    for gamma in [0.99, 0.997, 0.999]:
        add_job(
            "TVF_4L",
            run_name=f"baseline gamma={gamma}",
            env_name="Seaquest",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=False,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=10,
        )


    # test tvf_log_horizon, and new sampling rule
    # tvf_gamma = 0.997 # this helps ensure that shorter horizons matter more in terms of loss.
    # gamma = 0.999
    # for tvf_max_horizon in [1000, 2000, 4000]:
    #     add_job(
    #         "TVF_4M".format(env),
    #         run_name=f"tvf_mh={tvf_max_horizon}",
    #         env_name=env,
    #         checkpoint_every=int(5e6),
    #         epochs=50,
    #         agents=64,
    #         n_steps=512,
    #
    #         max_grad_norm=5.0,
    #         entropy_bonus=0.01,
    #
    #         use_tvf=True,
    #         tvf_coef=0.03,
    #         tvf_max_horizon=tvf_max_horizon,
    #         tvf_n_horizons=min(tvf_max_horizon, 250),
    #         tvf_gamma=tvf_gamma,
    #         tvf_advantage=True,
    #         tvf_distributional=True,
    #         tvf_log_horizon=True,
    #         vf_coef=0.0,
    #         tvf_epsilon=0.1,
    #
    #         workers=8,
    #         gamma=gamma,
    #         time_aware=False,
    #         priority=100,
    #     )


def setup_experiments5():

    # 5A Just to make sure we didn't break anything when we changed how sampling works etc
    tvf_gamma = 0.997  # this helps ensure that shorter horizons matter more in terms of loss.
    gamma = 0.999
    for tvf_max_horizon in [1000, 2000, 4000]:
        add_job(
            "TVF_5A",
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=250,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=20,
        )

    # 5B Horizon warmup
    tvf_gamma = 0.997  # this helps ensure that shorter horizons matter more in terms of loss.
    gamma = 0.999
    for tvf_max_horizon in [1000, 2000, 4000]:
        add_job(
            "TVF_5B",
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=250,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            tvf_horizon_warmup=0.1,

            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=0,
        )

    for tvf_max_horizon in [4000]:
        add_job(
            "TVF_5B",
            run_name=f"tvf_mh={tvf_max_horizon} tvf_coef=0.01",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.01,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=250,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            tvf_horizon_warmup=0.1,

            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=0,
        )


    for tvf_max_horizon in [4000]:
        add_job(
            "TVF_5B",
            run_name=f"tvf_mh={tvf_max_horizon} tvf_hw=0.5",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=250,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            tvf_horizon_warmup=0.5,

            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=0,
        )


    for tvf_max_horizon in [4000]:
        add_job(
            "TVF_5B",
            run_name=f"tvf_mh={tvf_max_horizon} tvf_hw=0.5 tvw_dist=linear",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=250,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            tvf_horizon_warmup=0.5,
            tvf_sample_dist="linear",

            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=0,
        )


    for tvf_max_horizon in [4000]:
        add_job(
            "TVF_5B",
            run_name=f"tvf_mh={tvf_max_horizon} distributional=False",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=250,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=False,
            tvf_horizon_warmup=0.1,

            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=0,
        )



    return

    # 5B Log horizon
    tvf_gamma = 0.997  # this helps ensure that shorter horizons matter more in terms of loss.
    gamma = 0.999
    for tvf_max_horizon in [1000, 2000, 4000]:
        add_job(
            "TVF_5B",
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=250,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,
            tvf_log_horizon=True,
            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=0,
        )

    # 5C Sample Dist
    tvf_gamma = 0.997  # this helps ensure that shorter horizons matter more in terms of loss.
    gamma = 0.999
    for tvf_max_horizon in [1000, 2000, 4000]:
        add_job(
            "TVF_5C",
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=250,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,

            tvf_sample_dist="linear",

            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=0,
        )

    # 5D TD Updates
    tvf_gamma = 0.997  # this helps ensure that shorter horizons matter more in terms of loss.
    gamma = 0.999
    for tvf_max_horizon in [1000, 2000, 4000]:
        add_job(
            "TVF_5D",
            run_name=f"tvf_mh={tvf_max_horizon}",
            env_name="Breakout",
            checkpoint_every=int(5e6),
            epochs=50,
            agents=64,
            n_steps=512,

            max_grad_norm=5.0,
            entropy_bonus=0.01,

            use_tvf=True,
            tvf_coef=0.03,
            tvf_max_horizon=tvf_max_horizon,
            tvf_n_horizons=250,
            tvf_gamma=tvf_gamma,
            tvf_advantage=True,
            tvf_distributional=True,

            tvf_lambda=0,

            vf_coef=0.0,
            tvf_epsilon=0.1,

            workers=8,
            gamma=gamma,
            time_aware=False,
            priority=0,
        )



def nice_format(x):
    if type(x) is str:
        return f'"{x}"'
    if type(x) in [int, float, bool]:
        return str(x)
    else:
        return f'"{x}"'


if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    id = 0
    job_list = []
    setup_mvh4()
    setup_experiments5()

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