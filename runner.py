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

OUTPUT_FOLDER = "/home/matthew/Dropbox/Experiments/ppo"

def add_job(experiment_name, run_name, priority=0, chunked=True, **kwargs):
    job_list.append(Job(experiment_name, run_name, priority, chunked, kwargs))

def get_run_folder(experiment_name, run_name):
    """ Returns the path for given experiment and run, or none if not found. """

    path = os.path.join(OUTPUT_FOLDER, experiment_name)
    if not os.path.exists(path):
        return None

    for file in os.listdir(path):
        name = os.path.split(file)[-1]
        this_run_name = name[:-19]  # crop off the id code.
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

        params_part = " ".join(["--{}='{}'".format(k, v) for k, v in self.params.items() if k not in ["env_name"]])
        params_part_lined = "\n".join(["--{}='{}'".format(k, v) for k, v in self.params.items() if k not in ["env_name"]])

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
            fps = details["fps"]
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


def setup_jobs_V7():

    # ------------------------------------------
    # RNN
    # ------------------------------------------

    # would be good to check mini-batch size, for performance I really want this up near 4k, instead of the 1k default.
    # for agents in [32]: # 32, 64, 128
    #     for n_step in [64, 128]: # 64, 128, 256
    #         for rnn_block_length in [0, 16, 32, 64]:
    #             use_rnn = (rnn_block_length != 0)
    #             add_job(
    #                 "RNN_Alien",
    #                 run_name="agents={} n_step={} block_length={}".format(agents, n_step, rnn_block_length),
    #                 env_name="Alien",
    #                 epochs=100,
    #                 agents=agents,
    #                 n_step=n_step,
    #                 rnn_block_length=rnn_block_length,
    #                 use_rnn = use_rnn,
    #                 frame_stack = 1 if use_rnn else 4 # frame stacking isn't needed with RNN
    #             )

    # ------------------------------------------
    # Diversity
    # ------------------------------------------

    # ok so computation is exploding here... here's what we're going to do.

    # primary run goes to 200M and we can do most of our analysis on that
    # to check for consistancy do 16 runs, but only up to 50M. Unfortunately I'll need this on pong and alien,
    # and I'll need it for all 3 modes. (I'll run 4 first, and do the others later on).
    # the idea is to show statistical significance for both the non-determanism, and later on for predicting sucessful
    # outcomes. (actually that's pong only...)


    # the old DIV_ experiments got cancelled due to noop start being inconsistant with Rainbow DQN. Also,
    # NONE trains on 0 noops but NOOP start never saw that, so it was difficult to compair them.
    # I'll keep the results around just in case I need them.

    # single runs of 4 games, with all 3 non-determanism
    for env in ["Pong", "Alien", "Seaquest", "MsPacman"]:
        run = 1
        for stochasticity in ["none", "noop", "sticky"]:
            add_job(
                "DIV2_{}".format(env),
                run_name="run={} stochasticity={}".format(run, stochasticity),
                env_name=env,
                export_trajectories=True,
                export_video=False ,         # this will take up too much space, and we can create them from the states anyway...
                checkpoint_every=int(1e6),   # every 1M is as frequent as possible (using current naming system)
                sticky_actions = stochasticity == "sticky",
                noop_start = stochasticity != "none",
                epochs=50, # maybe make this 200, or 100 or something later on?
                agents=32,
                priority=10
            )

    # 16 runs of alien, but only for 50M, and noop
    for env in ["Alien"]:
        stochasticity = "noop"
        for run in range(2,16+1):
            add_job(
                "DIV2_{}".format(env),
                run_name="run={} stochasticity={}".format(run, stochasticity),
                env_name=env,
                export_trajectories=True,
                export_video=False ,         # this will take up too much space, and we can create them from the states anyway...
                checkpoint_every=int(1e6),   # every 1M is as frequent as possible (using current naming system)
                sticky_actions = stochasticity == "sticky",
                noop_start = stochasticity != "none",
                epochs=50,
                agents=32,
                priority=0
            )

    # ------------------------------------------
    # Test gamma
    # ------------------------------------------

    # for gamma in [0.9, 0.99, 0.999, 0.9999, 0.99999, 1]:
    #     add_job(
    #         "Test_Gamma",
    #         run_name="gamma={}".format(gamma),
    #         env_name="SpaceInvaders",
    #         epochs=200,
    #         agents=32,
    #         priority=0
    #     )

    # ------------------------------------------
    # ARL
    # ------------------------------------------

    # for c_cost in [0.001, 0.01, 0.1]:
    #     for i_cost in [0.001, 0.01, 0.1]:
    #         add_job(
    #             "ARL_Breakout",
    #             env_name="Breakout",
    #             run_name="c_cost={} i_cost={}".format(c_cost, i_cost),
    #             arl_c_cost=c_cost,
    #             arl_i_cost=i_cost,
    #             algo="arl",
    #             epochs=100,
    #             priority=0,
    #             chunked=False
    #         )

    # ------------------------------------------
    # V-Trace
    # ------------------------------------------

    # # test algorithm on some games
    # for env in ["Alien", "Breakout"]:
    #     add_job(
    #         "VT_" + env,
    #         run_name="population=8 learning_rate=0.0003",
    #         learning_rate=3e-4, # slower is more stable...
    #         pbl_policy_soften=True,
    #         pbl_normalize_advantages="None",
    #         pbl_thinning="None",
    #         pbl_population_size=8,
    #         env_name=env,
    #         batch_epochs=2, # make sure we don't overtrain on the data. This also speeds up the training process.
    #         algo="pbl",
    #         epochs=200,
    #         agents=32,
    #         priority=10,
    #         chunked=False
    #     )
    #
    # # test algorithm on some games
    # add_job(
    #     "VT_Alien",
    #     run_name="population=8",
    #     learning_rate=1e-4,  # slower is more stable...
    #     pbl_policy_soften=True,
    #     pbl_normalize_advantages="None",
    #     pbl_thinning="None",
    #     pbl_population_size=8,
    #     env_name=env,
    #     batch_epochs=2,  # make sure we don't overtrain on the data. This also speeds up the training process.
    #     algo="pbl",
    #     epochs=200,
    #     agents=32,
    #     priority=10,
    #     chunked=False
    # )

if __name__ == "__main__":
    id = 0
    job_list = []
    setup_jobs_V7()

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