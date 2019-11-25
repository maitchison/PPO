import os
import sys
import shutil
import pandas as pd
import numpy as np
import json
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


OUTPUT_FOLDER = "/home/matthew/Dropbox/Experiments/ppo"

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

    def __init__(self, experiment_name, run_name, priority, params):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.priority = priority
        self.params = params
        self.id = Job.id
        Job.id += 1

    def __lt__(self, other):
         return self._sort_key < other._sort_key

    @property
    def _sort_key(self):
        return (-self.priority, self.experiment_name, self.id)

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
            last_modifed = os.path.getmtime(os.path.join(path, "params.txt"))

        if os.path.exists(os.path.join(path, "lock.txt")):
            status = "working"

        if os.path.exists(os.path.join(path, "final_score.txt")):
            status = "completed"

        if os.path.exists(os.path.join(path, "training_log.csv")):
            last_modifed = os.path.getmtime(os.path.join(path, "training_log.csv"))

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

        data = self.get_data()
        params = self.get_params()

        if data is None or params is None:
            return None

        details = {}
        details["max_epochs"] = params["epochs"]
        details["completed_epochs"] = data["Step"].iloc[-1] / 1e6

        scores = data["Ep_Score (100)"]
        scores = scores[~np.isnan(scores)]  #remove the nans
        if len(scores) > 10:
            details["score"] = np.percentile(scores, 95)
        else:
            details["score"] = 0

        details["fraction_complete"] = details["completed_epochs"] / details["max_epochs"]
        details["fps"] = np.mean(data["FPS"].iloc[-5:])
        frames_remaining = (details["max_epochs"] - details["completed_epochs"]) * 1e6
        details["eta"] = frames_remaining / details["fps"]
        details["host"] = params.get("hostname","unknown")

        return details

    def run(self, chunked=False):

        chunk_size = 10

        self.params["output_folder"] = OUTPUT_FOLDER

        experiment_folder = os.path.join(OUTPUT_FOLDER, self.experiment_name)
        src_folder = os.path.join(experiment_folder, "src")

        # make the destination folder...
        if not os.path.exists(experiment_folder):
            print("Making experiment folder " + experiment_folder)
            os.makedirs(experiment_folder, exist_ok=True)

        if not os.path.exists(src_folder):
            os.makedirs(src_folder, exist_ok=True)

        # copy script across if needed.
        ppo_path = os.path.join(src_folder, "ppo.py")
        if not os.path.exists(ppo_path):
            print("Copying ppo.py")
            shutil.copy("ppo.py", src_folder)

        self.params["experiment_name"] = self.experiment_name
        self.params["run_name"] = self.run_name

        details = self.get_details()

        if details is not None and details["completed_epochs"] > 0:
            # restore if some work has already been done.
            self.params["restore"] = True

        if chunked:
            # work out the next block to do
            if details is None:
                next_chunk = chunk_size
            else:
                next_chunk = (round(details["completed_epochs"] / chunk_size) * chunk_size) + chunk_size
            self.params["limit_epochs"] = int(next_chunk)


        python_part = "python {} {}".format(ppo_path, self.params["env_name"])

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


def add_job(experiment_name, run_name, priority=0, **kwargs):
    job_list.append(Job(experiment_name, run_name, priority, kwargs))


def setup_jobs():

    # -------------------------------------------------------------------------------------------
    # GA_Pong
    # -------------------------------------------------------------------------------------------

    for agents in reversed([8, 16, 32, 64, 128, 256, 512, 1024]):
        add_job(
            "GA_Pong",
            run_name="agents=" + str(agents),
            env_name="pong",
            epochs=20,
            agents=agents
        )

    for reward_clip in [1, 3, 5, 10]:
        add_job(
            "GA_Pong",
            run_name="reward_clip=" + str(reward_clip),
            env_name="pong",
            epochs=20,
            agents=256,
            reward_clip=reward_clip
        )

    for mini_batch_size in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        add_job(
            "GA_Pong",
            run_name="mini_batch_size="+str(mini_batch_size),
            env_name="pong",
            epochs=20,
            agents=256,
            mini_batch_size=mini_batch_size,
        )

    for mini_batch_size in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        agents = 64
        add_job(
            "GA_Pong",
            run_name="mini_batch_size={} agents={}".format(mini_batch_size, agents),
            env_name="pong",
            epochs=20,
            agents=64,
            mini_batch_size=mini_batch_size,
        )


    # -------------------------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------------------------

    add_job(
        "Test",
        run_name="test",
        env_name="pong",
        workers=32,
        epochs=200,
        agents=256,
        filter="hash",
        crop_input=True,
        learning_rate=1e-4,
        hash_size=7,
        priority=10
    )

    # -------------------------------------------------------------------------------------------
    # Hash
    # -------------------------------------------------------------------------------------------

    # get an idea of which hash size works...
    for hash_size in [1, 2, 4, 6, 7, 8, 16]:
        add_job(
            "Hash",
            run_name="hash_size=" + str(hash_size),
            env_name="pong",
            epochs=20,
            agents=64,
            filter="hash",
            hash_size=hash_size
        )

    for env_name in ["pong", "alien"]:
        for filter in ["hash", "hash_time"]:
            add_job(
                "Hash",
                run_name="Full {} {}".format(env_name, filter),
                env_name=env_name,
                epochs=200,
                agents=64,
                learning_rate=2e-4,
                filter=filter,
                hash_size=7,
                priority=5
            )

    add_job(
        "Hash",
        run_name="Full pong hash cropped",
        env_name="pong",
        epochs=200,
        agents=64,
        crop_input=True,
        learning_rate=2e-4,
        filter="hash",
        hash_size=7,
        priority=5
    )

    # -------------------------------------------------------------------------------------------
    # RA_Alien
    # -------------------------------------------------------------------------------------------

    for resolution in ["half", "standard", "full"]:
        for model in ["cnn", "cnn_improved"]:
            color = True if model == "improved_cnn" else False
            # kind of need to know the ideal params for this first...
            add_job(
                "RA_Alien",
                run_name="resolution={} model={} color={}".format(resolution, model, color),
                env_name="alien",
                epochs=20,
                agents=64,
                resolution=resolution,
                model=model,
                color=color
            )

def run_next_experiment(filter_jobs=None):

    job_list.sort()

    for job in job_list:
        if filter_jobs is not None and not filter_jobs(job):
            continue
        status = job.get_status()
        if status in ["pending", "waiting"]:
            job.run(chunked=True)
            return


def show_experiments(filter_jobs=None, all=False):
    job_list.sort()
    print("-" * 141)
    print("{:^10}{:<20}{:<60}{:>10}{:>10}{:>10}{:>10}{:>10}".format("priority", "experiment_name", "run_name", "complete", "status", "eta", "score", "host"))
    print("-" * 141)
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
            score = "{:.1f}".format(details["score"])
            host = details["host"][:8]
        else:
            percent_complete = ""
            eta_hours = ""
            score = ""
            host = ""


        status_transform = {
            "pending": "",
            "stale": "stale",
            "completed": "done",
            "working": "working",
            "waiting": "waiting"
        }

        print("{:^10}{:<20}{:<60}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
            job.priority, job.experiment_name, job.run_name, percent_complete, status_transform[status], eta_hours, score, host))

if __name__ == "__main__":
    id = 0
    job_list = []
    setup_jobs()

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