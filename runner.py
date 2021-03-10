import numpy
import os
import sys
import pandas as pd
import json
import time

from rl import utils

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

OUTPUT_FOLDER = "./Run"

def add_job(experiment_name, run_name, priority=0, chunked=True, **kwargs):
    if 'ignore_device' not in kwargs:
        kwargs['ignore_device'] = "[0]" # stub
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

        priority = priority - (self.get_completed_epochs() / 20)

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
            print("Making experiment folder " + experiment_folder)
            os.makedirs(experiment_folder, exist_ok=True)

        # copy script across if needed.
        train_script_path = utils.copy_source_files("./", experiment_folder)

        self.params["experiment_name"] = self.experiment_name
        self.params["run_name"] = self.run_name

        details = self.get_details()

        if details is not None and details["completed_epochs"] > 0:
            # restore if some work has already been done.
            self.params["restore"] = True

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

    # experiment 1
    for env in ["Breakout"]:
        for heads in [1, 3, 5, 10]:
            for prior in [False]:
                if prior and heads == 0:
                    continue
                add_job(
                    "mvh_{}".format(env),
                    run_name=f"run={env} prior={prior} heads={heads}",
                    env_name=env,
                    checkpoint_every=int(10e6),  # every 1M is as frequent as possible (using current naming system)
                    epochs=100,
                    agents=256,
                    workers=8,
                    use_mvh=heads > 0,
                    mvh_heads=heads,
                    mvh_prior=prior,
                )


    # experiment 2
    for env in ["Breakout"]:
        for heads in [1, 3, 5, 10]:
            for prior in [False, True]:
                add_job(
                    "MVH2".format(env),
                    run_name=f"run={env} prior={prior} heads={heads}",
                    env_name=env,
                    checkpoint_every=int(10e6),  # every 1M is as frequent as possible (using current naming system)
                    epochs=50,
                    agents=256,
                    workers=8,
                    use_mvh=True,
                    mvh_heads=heads,
                    mvh_prior=prior,
                )

    # gamma test
    for env in ["Breakout"]:
        for gamma in [0.9, 0.99, 0.999, 0.9999, 0.99999, 1.0]:
            add_job(
                "GAMMA".format(env),
                run_name=f"gamma={gamma}",
                env_name=env,
                checkpoint_every=int(10e6),  # every 1M is as frequent as possible (using current naming system)
                epochs=100,
                agents=256,
                workers=8,
                use_mvh=False,
                gamma=gamma,
            )

    # td_gamma test
    # abs error
    for env in ["Breakout"]:
        for td_gamma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
            add_job(
                "TD_GAMMA".format(env),
                run_name=f"td_gamma={td_gamma}",
                env_name=env,
                checkpoint_every=int(10e6),
                epochs=100,
                agents=256,
                workers=8,
                use_mvh=False,
                td_gamma=td_gamma,
            )

    # td_gamma test
    # squared error
    for env in ["Breakout"]:
        for td_gamma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
            add_job(
                "TD_GAMMA2".format(env),
                run_name=f"td_gamma={td_gamma}",
                env_name=env,
                checkpoint_every=int(10e6),
                epochs=50,
                agents=256,
                workers=8,
                use_mvh=False,
                td_gamma=td_gamma,
            )

    # td_gamma test
    # just error
    for env in ["Breakout"]:
        for td_gamma in [-1, -0.1, -0.3, 0, 0.1, 0.3, 1]:
            add_job(
                "TD_GAMMA3".format(env),
                run_name=f"td_gamma={td_gamma}",
                env_name=env,
                checkpoint_every=int(10e6),
                epochs=50,
                agents=256,
                workers=8,
                use_mvh=False,
                td_gamma=td_gamma,
            )

    # episodic discounting
    for env in ["Breakout"]:
        for gamma in [1.0, 0.999, 0.99]:
            for ed_gamma in [1.0, 0.999, 0.99]:
                for ed_type in ["geometric", "hyperbolic"]:

                    add_job(
                        "ED1".format(env),
                        run_name=f"ed_gamma={ed_gamma} ed_type={ed_type} gamma={gamma}",
                        env_name=env,
                        checkpoint_every=int(10e6),
                        epochs=50,
                        agents=256,
                        workers=8,

                        time_aware=True,
                        ed_type=ed_type,
                        ed_gamma=ed_gamma,
                        gamma=gamma,

                        use_mvh=False,
                    )

    # gamma test with new updated GAE returns calculation
    for env in ["Breakout"]:
        for gamma in [0.9, 0.99, 0.999, 0.9999, 1.0]:
            add_job(
                "GAE_GAMMA".format(env),
                run_name=f"gamma={gamma}",
                env_name=env,
                checkpoint_every=int(10e6),  # every 1M is as frequent as possible (using current naming system)
                epochs=50,
                agents=256,
                workers=8,
                gamma=gamma,
            )

    # gamma test with new updated GAE returns calculation and time aware
    for env in ["Breakout"]:
        for gamma in [0.9, 0.99, 0.999, 0.9999, 1.0]:
            add_job(
                "TVF_1C".format(env),
                run_name=f"gamma={gamma}",
                env_name=env,
                checkpoint_every=int(10e6),  # every 1M is as frequent as possible (using current naming system)
                epochs=50,
                agents=256,
                workers=8,
                gamma=gamma,
                time_aware=True,
            )

    # truncated value function as aux task
    for env in ["Breakout"]:
        for gamma in [0.99, 0.999, 0.9999]:
            add_job(
                "VFH_AUX_100".format(env),
                run_name=f"gamma={gamma}",
                env_name=env,
                checkpoint_every=int(10e6),  # every 1M is as frequent as possible (using current naming system)
                epochs=50,
                agents=256,
                use_vfh=True,
                workers=8,
                gamma=gamma,
            )

    # truncated value function as aux task (reduced coef)
    for env in ["Breakout"]:
        for gamma in [0.99, 0.999, 0.9999]:
            add_job(
                "TVF_100_AUX".format(env),
                run_name=f"gamma={gamma}",
                env_name=env,
                checkpoint_every=int(10e6),  # every 1M is as frequent as possible (using current naming system)
                epochs=50,
                agents=256,
                use_tvf=True,
                tvf_coef=0.1,
                workers=8,
                gamma=gamma,
                time_aware=True,
            )


    # truncated value function as aux task fixed (hopefuly) targets)
    for env in ["Breakout"]:
        for gamma in [0.99, 0.999, 0.9999]:
            add_job(
                "TVF_2D".format(env),
                run_name=f"gamma={gamma}",
                env_name=env,
                checkpoint_every=int(5e6),  # every 1M is as frequent as possible (using current naming system)
                epochs=50,
                agents=256,
                use_tvf=True,
                tvf_coef=0.1,
                workers=8,
                gamma=gamma,
                time_aware=False,
            )

    # truncated value function as aux task longer horizon, higher gamma
    for env in ["Breakout"]:
        for gamma in [0.99, 0.999, 0.9999]:
            add_job(
                "TVF_2E".format(env),
                run_name=f"gamma={gamma}",
                env_name=env,
                checkpoint_every=int(5e6),  # every 1M is as frequent as possible (using current naming system)
                epochs=50,
                agents=256,

                use_tvf=True,
                tvf_coef=0.1,
                tvf_horizon=1000,
                tvf_gamma=0.999,

                workers=8,
                gamma=gamma,
                time_aware=False,
                priority=10,
            )

    # make sure random sampling is working
    # for env in ["Breakout"]:
    #     for n_steps in [16, 64, 128]:
    #         for gamma in [0.99]:
    #             for tvf_gamma in [0.99]:
    #                 for tvf_max_horizon in [30, 100, 300]:
    #                     for tvf_n_horizons in [10, 30, 100]:
    #                         if tvf_n_horizons > tvf_max_horizon:
    #                             continue
    #                         add_job(
    #                             "TVF_2F".format(env),
    #                             run_name=f"ns={n_steps} g={gamma} tg={tvf_gamma} tmh={tvf_max_horizon} tnh={tvf_n_horizons}",
    #                             env_name=env,
    #                             checkpoint_every=int(5e6),  # every 1M is as frequent as possible (using current naming system)
    #                             epochs=50,
    #                             agents=256,
    #                             n_steps=n_steps,
    #
    #                             use_tvf=True,
    #                             tvf_coef=0.1,
    #                             tvf_max_horizon=tvf_max_horizon,
    #                             tvf_n_horizons=tvf_n_horizons,
    #                             tvf_gamma=tvf_gamma,
    #
    #                             priority=-10,
    #                             workers=8,
    #                             gamma=gamma,
    #                             time_aware=False,
    #                         )

    # truncated value function as aux task longer horizon, higher gamma
    for env in ["Breakout"]:
        for gamma in [0.99]:
            add_job(
                "TVF_2G".format(env),
                run_name=f"gamma={gamma}",
                env_name=env,
                checkpoint_every=int(5e6),
                epochs=100,
                agents=256,

                use_tvf=True,
                tvf_coef=0.1,
                tvf_max_horizon=300,
                tvf_n_horizons=100,
                tvf_gamma=0.99,

                workers=8,
                gamma=gamma,
                time_aware=False,
                priority=10,
            )

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
    setup_mvh()

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
