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


def add_job(experiment_name, run_name, priority=0, **kwargs):
    job_list.append(Job(experiment_name, run_name, priority, kwargs))


def setup_jobs_V5():

    # regression testing
    for env_name in ["Pong", "Seaquest", "Breakout", "Alien", "MonetzumaRevenge"]:
        add_job(
            "Regression "+env_name,
            run_name="Regression v0.5",
            env_name=env_name,
            epochs=200 if env_name != "Pong" else 50,
            priority=0
        )

    for movement_cost in [0, 0.2, 0.02]:
        for global_frame_skip in [1, 10]:
            add_job(
                "Fovea",
                run_name="global_frame_skip={} movement_cost={}".format(global_frame_skip, movement_cost),
                env_name="Pong",
                use_atn=True,
                atn_global_frame_skip=global_frame_skip,
                atn_movement_cost=movement_cost,
                epochs=30,
                agents=64,
                priority=0
            )

    # try to find some better hyperparameters
    for extrinsic_reward_scale in [1.0, 2.0, 4.0]:
        add_job(
            "EXP_RND_v7",
            run_name="RND ext_rew={}".format(extrinsic_reward_scale),
            env_name="MontezumaRevenge",
            epochs=200,
            agents=64,
            n_steps=128,
            entropy_bonus=0.001,
            learning_rate=1e-4,
            mini_batch_size=1024,  # seems very high!
            gae_lambda=0.95,
            ppo_epsilon=0.1,
            gamma=0.995,
            gamma_int=0.99,
            sticky_actions=True,
            max_grad_norm=5,  # they really do set this to 0...
            reward_normalization=True,
            noop_start=True,
            adam_epsilon=1e-5,
            intrinsic_reward_scale=1.0,
            extrinsic_reward_scale=extrinsic_reward_scale,
            normalize_advantages=True,
            use_clipped_value_loss=True,

            use_rnd=True,
            priority=0
        )

    # this one worked well before
    add_job(
        "EXP_RND_v7",
        run_name="RND prior_best",
        env_name="MontezumaRevenge",
        epochs=200,
        agents=32,
        n_steps=128,
        entropy_bonus=0.001,  # why slow low?
        learning_rate=1e-4,
        mini_batch_size=1024,  # seems very high!
        gae_lambda=0.95,
        ppo_epsilon=0.1,
        gamma=0.995,
        gamma_int=0.99,
        sticky_actions=True,
        max_grad_norm=0,  # they really do set this to 0...
        reward_normalization=False,
        noop_start=False,
        reward_clip=1,
        adam_epsilon=1e-8,  # they use the default in TF which is 1e-8
        intrinsic_reward_scale=1.0,
        extrinsic_reward_scale=2.0,
        normalize_advantages=False,
        use_clipped_value_loss=False,

        use_rnd=True,
        priority=2
    )


def setup_jobs_V4():

    # simple regression test, make sure agent can learn the problem and that normalization constants get saved
    # and restored properly.
    add_job(
        "Test",
        run_name="Pong",
        env_name="Pong",
        epochs=20,
        agents=128,
        priority=20
    )

    # -------------------------------------------------------------------------------------------
    # Memorization 2
    # -------------------------------------------------------------------------------------------

    for memorize_cards in [1, 10, 100]:
        for memorize_actions in [2, 4, 8]:
            for model_hidden_units in [1, 2, 4, 8, 16]:
                add_job(
                    "Memorize_Units",
                    run_name="cards={} actions={} hidden_units={}".format(memorize_cards, memorize_actions, model_hidden_units),
                    env_name="Memorize",
                    model_hidden_units=model_hidden_units,
                    agents=16,
                    epochs=1,
                    n_steps=64,
                    memorize_cards=memorize_cards,
                    memorize_actions=memorize_actions,
                    priority=7
                )

    # -------------------------------------------------------------------------------------------
    # Freeze Layers
    # -------------------------------------------------------------------------------------------

    for freeze_layers in [0,1,2,3,4]:
        add_job(
            "Freeze_Layers",
            run_name="freeze_layers={}".format(freeze_layers),
            env_name="Pong",
            epochs=100 if freeze_layers in [3,4] else 50,
            agents=32,
            freeze_layers=freeze_layers,
            priority=5
        )
    # -------------------------------------------------------------------------------------------
    # EXP_RND
    # -------------------------------------------------------------------------------------------

    # this one worked well before
    add_job(
        "EXP_RND_v6",
        run_name="RND repeat best",
        env_name="MontezumaRevenge",
        epochs=200,
        agents=32,
        n_steps=128,
        entropy_bonus=0.001,  # why slow low?
        learning_rate=2.5e-4,
        mini_batch_size=1024,  # seems very high!
        gae_lambda=0.95,
        ppo_epsilon=0.1,
        gamma=0.99,
        gamma_int=0.99,
        sticky_actions=True,
        max_grad_norm=0,  # they really do set this to 0...
        reward_normalization=False,
        noop_start=False,
        reward_clip=1,
        adam_epsilon=1e-8,  # they use the default in TF which is 1e-8
        extrinsic_reward_scale=2.0,
        normalize_advantages=False,
        use_clipped_value_loss=True,

        use_rnd=True,
        priority=25
    )

    # Reproduction study on RND paper.
    for ext_rew_scale in [0.5, 1.0, 2.0]:
        add_job(
            "EXP_RND_v6",
            run_name="RND ext_rew_scale={}".format(ext_rew_scale),
            env_name="MontezumaRevenge",
            epochs=200,
            agents=32,
            n_steps=128,
            entropy_bonus=0.001,          # why slow low?
            learning_rate=1e-4,
            mini_batch_size=1024,         # seems very high!
            gae_lambda=0.95,
            ppo_epsilon=0.1,
            gamma=0.995,
            gamma_int=0.99,
            sticky_actions=True,
            max_grad_norm=0,              # they really do set this to 0...
            reward_normalization=False,
            noop_start=False,
            reward_clip=1,
            adam_epsilon=1e-8,            # they use the default in TF which is 1e-8
            extrinsic_reward_scale=ext_rew_scale,
            normalize_advantages=False,
            use_clipped_value_loss=False,

            use_rnd=True,
            priority=25 if ext_rew_scale == 2.0 else 18
        )

    add_job(
        "EXP_RND_v6",
        run_name="RND (alternative)",
        env_name="MontezumaRevenge",
        epochs=200,
        agents=32,
        n_steps=128,
        entropy_bonus=0.001,  # why slow low?
        learning_rate=1e-4,
        mini_batch_size=1024,  # seems very high!
        gae_lambda=0.95,
        ppo_epsilon=0.1,
        gamma=0.995,
        gamma_int=0.99,
        sticky_actions=True,
        max_grad_norm=1,  # they really do set this to 0...
        reward_normalization=True,
        noop_start=False,
        adam_epsilon=1e-5,  # they use the default in TF which is 1e-8
        extrinsic_reward_scale=2.0,
        normalize_advantages=True,
        use_clipped_value_loss=True,

        use_rnd=True,
        priority=20
    )

    # Reproduction study on RND paper.
    for int_rew_scale in [0.5, 1.0, 2.0]:
        add_job(
            "EXP_RND_v5",
            run_name="RND int_rew_scale={}".format(int_rew_scale),
            env_name="MontezumaRevenge",
            epochs=20,
            agents=32,
            n_steps=128,
            entropy_bonus=0.001,  # why slow low?
            learning_rate=1e-4,
            mini_batch_size=1024,  # seems very high!
            gae_lambda=0.95,
            ppo_epsilon=0.1,
            gamma=0.99,  # separate gamma not working yet..
            gamma_int=0.99,
            sticky_actions=True,
            max_grad_norm=1,
            reward_normalization=False,
            noop_start=False,
            reward_clip=1,
            adam_epsilon=1e-8,  # they used 1e-8 :( seems high...
            intrinsic_reward_scale=int_rew_scale,

            use_rnd=True,
            priority=10

        )

    # Reproduction study on RND paper.
    for int_rew_scale in [0.5, 1.0, 2.0]:
        add_job(
            "EXP_RND_v4",
            run_name="RND int_rew_scale={}".format(int_rew_scale),
            env_name="MontezumaRevenge",
            epochs=20,
            agents=32,
            n_steps=128,
            entropy_bonus=0.001,  # why slow low?
            learning_rate=1e-4,
            mini_batch_size=1024,  # seems very high!
            gae_lambda=0.95,
            ppo_epsilon=0.1,
            gamma=0.99,                             # separate gamma not working yet..
            gamma_int=0.99,
            sticky_actions=True,
            max_grad_norm=1,
            reward_normalization=False,
            noop_start=False,
            reward_clip=1,
            adam_epsilon=1e-5,                      # they used 1e-8 :( seems high...
            intrinsic_reward_scale=int_rew_scale,

            use_rnd=True,
            priority=10

        )

    # Reproduction study on RND paper.
    for int_rew_scale in [0.25, 0.5, 1.0]:
        add_job(
            "EXP_RND_v3",
            run_name="RND Third Test int_rew_scale={}".format(int_rew_scale),
            env_name="MontezumaRevenge",
            epochs=25,
            agents=32,
            n_steps=128,
            entropy_bonus=0.001,  # why slow low?
            learning_rate=1e-4,
            mini_batch_size=1024,  # seems very high!
            gae_lambda=0.95,
            ppo_epsilon=0.1,
            gamma=0.99,  # separate gamma not working yet..
            gamma_int=0.99,
            sticky_actions=True,
            max_grad_norm=5,  # they use 0... but... nope...
            reward_normalization=True,  # this just make a lot more sense to me
            adam_epsilon=1e-5,  # they used 1e-8.. but nope...
            intrinsic_reward_scale=int_rew_scale,

            use_rnd=True,
            priority=1

        )

    # Just make sure pong actually works using these settings
    add_job(
        "EXP_RND_v3",
        run_name="RND Third Test (pong)",
        env_name="Pong",
        epochs=20,
        agents=32,
        n_steps=128,
        entropy_bonus=0.001,  # why slow low?
        learning_rate=1e-4,
        mini_batch_size=1024,  # seems very high!
        gae_lambda=0.95,
        ppo_epsilon=0.1,
        gamma=0.99,     # separate gamma not working yet..
        gamma_int=0.99,
        sticky_actions=True,
        max_grad_norm=5,  # they use 0... but... nope...
        reward_normalization=True,  # this just make a lot more sense to me
        adam_epsilon=1e-5,  # they used 1e-8.. but nope...
        intrinsic_reward_scale=1.0,

        use_rnd=True,
        priority=1

    )

    # this one should get close to 2,500 after 50M steps.
    add_job(
        "EXP_RND_v3",
        run_name="PPO Baseline (2)",
        env_name="MontezumaRevenge",
        epochs=40,
        agents=32,
        n_steps=128,
        entropy_bonus=0.001,
        learning_rate=1e-4,
        mini_batch_size=1024,
        gae_lambda=0.95,
        ppo_epsilon=0.1,
        gamma=0.99,
        sticky_actions=True,
        max_grad_norm=5,  # taken from source code.
        adam_epsilon=1e-5,

        use_rnd=False,
        priority=1
    )


def setup_jobs_V3():

    # -------------------------------------------------------------------------------------------
    # GA_Pong
    # -------------------------------------------------------------------------------------------

    for agents in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        add_job(
            "GA_Pong_Agents",
            run_name="agents=" + str(agents),
            env_name="pong",
            epochs=20,
            agents=agents,
            priority=10,
        )

    for reward_clip in [1, 3, 5, 10]:
        add_job(
            "GA_Pong_Reward_Clip",
            run_name="reward_clip=" + str(reward_clip),
            env_name="pong",
            epochs=20,
            agents=256,
            reward_clip=reward_clip
        )

    for max_grad_norm in [0.05, 0.5, 5, 50, 0]:
        for agents in [64, 256]:
            add_job(
                "GA_Pong_Max_Grad_Norm",
                run_name="max_grad_norm={} agents={}".format(max_grad_norm, agents),
                env_name="pong",
                epochs=20,
                agents=agents,
                max_grad_norm=max_grad_norm,
                priority=1
            )

    for mini_batch_size in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        for agents in [64, 256]:
            add_job(
                "GA_Pong_Mini_Batch",
                run_name="mini_batch_size={} agents={}".format(mini_batch_size, agents),
                env_name="pong",
                epochs=20,
                agents=agents,
                mini_batch_size=mini_batch_size,
            )


    # -------------------------------------------------------------------------------------------
    # GA_Alien
    # -------------------------------------------------------------------------------------------

    # just get a quick performance check on alien.
    for agents in [64, 256, 1024]:
        for n_steps in [8, 32, 64, 128]:
            mini_batch_size = 1024
            add_job(
                "GA_Alien",
                run_name="agents={} n_steps={} mini_batch_size={}".format(agents, n_steps, mini_batch_size),
                env_name="Alien",
                epochs=200,
                mini_batch_size=mini_batch_size,
                agents=agents,
                n_steps=n_steps
            )

    # -------------------------------------------------------------------------------------------
    # GA_Seaquest
    # -------------------------------------------------------------------------------------------

    # just get a quick performance check on breakout.
    add_job(
        "GA_Seaquest",
        run_name="baseline",
        env_name="Seaquest",
        epochs=200,
        agents=64,
    )


    # -------------------------------------------------------------------------------------------
    # GA_Breakout
    # -------------------------------------------------------------------------------------------

    # just get a quick performance check on breakout.
    add_job(
        "GA_Breakout",
        run_name="baseline",
        env_name="Breakout",
        epochs=200,
        agents=64,
    )

    # -------------------------------------------------------------------------------------------
    # GA_MontezumaRevenge
    # -------------------------------------------------------------------------------------------

    # just get a quick performance check on montezuma's revenge.
    add_job(
        "GA_MontezumaRevenge",
        run_name="baseline",
        env_name="MontezumaRevenge",
        epochs=200,
        agents=64,
    )

    # -------------------------------------------------------------------------------------------
    # Memorization
    # -------------------------------------------------------------------------------------------

    for memorize_cards in [1, 10, 100, 300, 1000, 3000, 10000]:
        add_job(
            "Memorize",
            run_name="cards={}".format(memorize_cards),
            env_name="Memorize",
            agents=16,
            epochs=10,
            n_steps=64,
            memorize_cards=memorize_cards,
            priority=2
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
                run_name="full {} {}".format(env_name, filter),
                env_name=env_name,
                epochs=100,
                agents=64,
                learning_rate=2e-4,
                filter=filter,
                hash_size=7,
            )

    add_job(
        "Hash",
        run_name="full pong hash cropped",
        env_name="pong",
        epochs=100,
        agents=64,
        input_crop=True,
        learning_rate=1e-4,
        filter="hash",
        hash_size=7,
        priority=12,
    )

    # -------------------------------------------------------------------------------------------
    # RA_Alien
    # -------------------------------------------------------------------------------------------

    """
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
    """

def run_next_experiment(filter_jobs=None):

    job_list.sort()

    for job in job_list:
        if filter_jobs is not None and not filter_jobs(job):
            continue
        status = job.get_status()
        if status in ["pending", "waiting"]:
            job.run(chunked=True)
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

if __name__ == "__main__":
    id = 0
    job_list = []
    #setup_jobs_V4()
    setup_jobs_V5()

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