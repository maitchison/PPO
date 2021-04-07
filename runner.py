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


CHUNK_SIZE=10
DEVICE="auto"
OUTPUT_FOLDER="./Run"
WORKERS=8

if len(sys.argv) == 3:
    DEVICE = sys.argv[2]

def add_job(experiment_name, run_name, priority=0, chunked=True, default_params=None, **kwargs):

    if default_params is not None:
        for k,v in default_params.items():
            if k not in kwargs:
                kwargs[k]=v

    if "device" not in kwargs:
        kwargs["device"] = DEVICE



    job_list.append(Job(experiment_name, run_name, priority, chunked, kwargs))

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
        if status == "running":
            priority += 1000

        # if "search" in self.experiment_name:
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

        if status in ["", "pending"]:

            job.get_params()

            job.run(chunked=job.chunked)
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

def random_search(run, main_params, search_params, count=128):

    for i in range(count):
        params = {}
        random.seed(i)
        for k, v in search_params.items():
            params[k] = random.choice(v)

        # make sure params arn't too high (due to memory)
        while params["agents"] * params["n_steps"] > 64*1024:
            params["agents"] //= 2

        add_job(run, run_name=f"{i:04d}", chunked=True, **main_params, **params)


def nice_format(x):
    if type(x) is str:
        return f'"{x}"'
    if x is None:
        return "None"
    if type(x) in [int, float, bool]:
        return str(x)

    return f'"{x}"'

# ---------------------------------------------------------------------------------------------------------


def setup_experiments9():    

    # these are just for the regression test
    initial_args = {
        'checkpoint_every': int(5e6),
        'epochs': 50,
        'agents': 256,
        'n_steps': 128,
        'max_grad_norm': 5.0,
        'entropy_bonus': 0.01,
        'use_tvf': True,
        'tvf_coef': 0.01, # because of new scaling this needs to be 0.1 rather than 0.5 to avoid high grad at start.
        'vf_coef': 0.0,
        'tvf_max_horizon': 1000,
        'tvf_n_horizons': 64,
        'workers': WORKERS,
        'tvf_gamma': 0.997,
        'gamma': 0.997,
        'mini_batch_size': 1024,
        'tvf_model': 'split',
        'joint_model_weight': 0.1, # seems like a small amount of this is helpful for breakout
        'tvf_joint_mode': 'policy', # this seems to work best
    }

    # these are just for the regression test
    ppo_args = {
        'checkpoint_every': int(5e6),
        'epochs': 50,
        'agents': 256,
        'n_steps': 128,
        'max_grad_norm': 5.0,
        'entropy_bonus': 0.01,
        'use_tvf': False,
        'vf_coef': 0.5,
        'workers': WORKERS,
        'gamma': 0.997,
        'mini_batch_size': 1024,
    }

    # Standard regression test
    for tvf_max_horizon in [1000, 2000, 4000]:
        add_job(
            "TVF_9A",
            env_name="Breakout",
            run_name=f"tvf_mh={tvf_max_horizon}",
            tvf_max_horizon=tvf_max_horizon,
            priority=20,
            default_params=initial_args,
        )

    # Test reward clipping change
    for reward_clipping in [1, "off", "sqrt"]:
        for reward_normalization in [True, False]:
            add_job(
                "TVF_9_RewardClipping",
                env_name="DemonAttack",
                run_name=f"norm={reward_normalization} clip={reward_clipping}",
                reward_clipping=reward_clipping,
                reward_normalization=reward_normalization,
                priority=0,
                default_params=initial_args,
            )

    # Test gamma
    for gamma in [0.99, 0.997, 0.999, 0.9999, 1.0]:
        add_job(
            "TVF_9_Gamma",
            env_name="DemonAttack",
            run_name=f"tvf_3k gamma={gamma}",
            tvf_max_horizon=3000,
            gamma=gamma,
            tvf_gamma=gamma,
            priority=0,
            default_params=initial_args,
        )

        add_job(
            "TVF_9_Gamma",
            env_name="DemonAttack",
            run_name=f"ppo gamma={gamma}",
            gamma=gamma,
            priority=0,
            default_params=ppo_args,
        )

    # Test deferred rewards (very hard!)
    add_job(
        "TVF_9_DeferredReward",
        env_name="DemonAttack",
        run_name=f"algo=tvf",
        deferred_rewards=True,
        priority=0,
        default_params=initial_args,
    )

    # Test deferred rewards (very hard!)
    add_job(
        "TVF_9_DeferredReward",
        env_name="DemonAttack",
        run_name=f"algo=tvf_3k",
        tvf_max_horizon=3000,
        deferred_rewards=True,
        priority=0,
        default_params=initial_args,
    )

    # Test deferred rewards (very hard!)
    add_job(
        "TVF_9_DeferredReward",
        env_name="DemonAttack",
        run_name=f"algo=tvf_3k_nd",
        tvf_gamma=1.0,
        gamma=1.0,
        tvf_max_horizon=3000,
        deferred_rewards=True,
        priority=25,
        default_params=initial_args,
    )

    # Test deferred rewards (very hard!)
    add_job(
        "TVF_9_DeferredReward",
        env_name="DemonAttack",
        run_name=f"algo=ppo",
        deferred_rewards=True,
        priority=0,
        default_params=ppo_args,
    )

    add_job(
        "TVF_9_DeferredReward",
        env_name="DemonAttack",
        run_name=f"algo=ppo_nd",
        deferred_rewards=True,
        gamma=1.0,
        priority=50,
        default_params=ppo_args,
    )

    search_vars = {
        'n_steps':[4, 8, 16, 32, 64, None, 256, 512, 1024],
        'agents': [32, 64, 128, None, 512, 1024],
        'tvf_max_horizon': [300, None, 3000, 10000],
        'use_training_pauses': [True, None],
        'tvf_hidden_units': [0, 1, 4, 64, 128, 256, None, 1024],
        'tvf_n_horizons': [4, 8, 16, 32, None, 128, 1000], # 1000 has no sampling (we have 2 compulsary horizons so 4 is very low!
        'max_grad_norm': [0.5, None, 20.0],
        'learning_rate': [1e-4, None, 1e-3],
        'entropy_bonus': [0, 0.001, None, 0.1],         # 0.01 was default
        'joint_model_weight': [-1, 0, 0.01, None, 1],   # 0.1 was default
        'tvf_model': ['default', None],
        'tvf_coef': [0.05, None, 0.5],
        'tvf_loss_weighting': [None, 'advanced'],
        'mini_batch_size': [512, None, 2048],
        'ppo_epsilon': [0.05, None, 0.2, 0.3], # this should have been 0.2 by default
        'tvf_gamma': [1.0, None],  # try rediscounting again... might give better estimates (but slower...)
        'batch_epochs': [2, 3, None, 6, 8], # this makes a big difference
        'tvf_h_scale': [None, 'linear', 'squared'],
        'tvf_activation': [None, 'tanh', 'sigmoid'],
        'max_micro_batch_size': [256, None, 1024], #just checking...
        'sticky_actions': [True, None],  # why not...
        'color': [True, None],  # not even sure if this works
        'resolution': ['full', None, 'half'],  # again not sure if this works...
        'per_step_reward': [None, round(-1 / (60 * 30 * 15), 6)], # encourage agent to complete game in a reasonable time.
    }

    counter = 0
    for env in ['DemonAttack']:
        for k, vs in search_vars.items():
            for v in vs:
                if v is None:
                    # these represent the default settings, no need to run these as I have run them 3 times already...
                    continue
                add_job(
                    f"TVF_9_Search_{env}",
                    run_name=f"{k}={v}",
                    **{k: v},
                    env_name=env,
                    epochs=20, # should be 50, but just for the moment..
                    priority=-(100+counter),
                    default_params=initial_args,
                )
                counter += 1


    # initial tests on our new games
    #for env in ['Amidar', 'BattleZone', 'DemonAttack']:
    for env in ['DemonAttack']:
        for run in range(3):
            add_job(
                f"TVF_9_Ref_{env}",
                run_name=f"tvf run={run}",
                env_name=env,
                priority=0 if run != 0 else 100,
                default_params=initial_args,
            )

    # just some games I'd be interested in seeing the score on.
    for env in ['Alien', 'Skiing', 'Seaquest', 'Pitfall']:
        for run in range(1):
            add_job(
                f"TVF_9_Ref_{env}",
                run_name=f"tvf run={run}",
                env_name=env,
                priority=-100 if env != "Skiing" else 100,
                default_params=initial_args,
            )
        add_job(
            f"TVF_9_Ref_{env}",
            run_name=f"tvf_rnd",
            env_name=env,
            use_rnd=True,
            observation_normalization=True,
            vf_coef=0.5,
            intrinsic_reward_scale=1.0,
            priority=-150,
            default_params=initial_args,
        )

    add_job(
        f"TVF_9_Ref_Skiing",
        run_name=f"tvf_5k_rnd",
        env_name='Skiing',
        use_rnd=True,
        observation_normalization=True,
        vf_coef=0.5,
        intrinsic_reward_scale=1.0,
        tvf_max_horizon=5000,
        priority=100,
        default_params=initial_args,
    )

    add_job(
        f"TVF_9_Ref_Skiing",
        run_name=f"tvf_5k_rnd_nd",
        env_name='Skiing',
        use_rnd=True,
        observation_normalization=True,
        vf_coef=0.5,
        intrinsic_reward_scale=1.0,
        tvf_max_horizon=5000,
        priority=100,
        tvf_gamma=1.0,
        gamma=1.0,
        default_params=initial_args,
    )

    add_job(
        f"TVF_9_Ref_Skiing",
        run_name=f"tvf_5k",
        env_name='Skiing',
        tvf_max_horizon=5000,
        priority=100,
        default_params=initial_args,
    )

    add_job(
        f"TVF_9_Ref_Skiing",
        run_name=f"ppo",
        env_name='Skiing',

        priority=100,
        default_params=ppo_args,
    )

    # this is just best settings found so far at various points
    for env in ['DemonAttack']:
        add_job(
            f"TVF_9_Eval_{env}",
            run_name=f"bundle_0",
            env_name=env,

            epochs=200,

            priority=0,
            default_params=initial_args,
        )

        add_job(
            f"TVF_9_Eval_{env}_1",
            run_name=f"bundle_1",
            env_name=env,
            n_steps=32,                 # 16 was a little better, but we've upped max_horizon
            agents=256,
            tvf_max_horizon=3000,
            reward_clipping="sqrt",

            epochs=200,

            priority=50,
            default_params=initial_args,
        )

        # bundle 2 ideas
        # max_horizon to 1000 for performance reasons
        # turn of sqrt clipping, it's not needed (less parts)
        # agents=128 will be fine, reduces memory
        # hidden units=256 will be faster, less GPU ram
        # keep n_horizons at 64, as it's fast enough


def setup_experiments10():

    # these are just for the regression test
    initial_args = {
        'checkpoint_every': int(5e6),
        'env_name': 'DemonAttack',
        'epochs': 50,
        'agents': 256,
        'n_steps': 256,
        'max_grad_norm': 5.0,
        'entropy_bonus': 0.01,
        'use_tvf': True,
        'tvf_max_horizon': 1000,
        'tvf_n_horizons': 64,
        'tvf_coef': 0.01,
        'vf_coef':0.5,              # this won't do anything (by default)
        'workers': WORKERS,
        'tvf_gamma': 0.999,
        'gamma': 0.999,
        'policy_mini_batch_size': 2048,
        'value_mini_batch_size': 256,
    }

    # this is very close to the old settings
    # this worked **very** well, I should probably use this as the reference, and see if I can get TVF to perform
    # as well as this. Maybe we can improve things a little by tweaking too?
    add_job(
        f"TVF_10_Regression",
        run_name=f"ppo (alt)",
        max_grad_norm=20.0,
        agents=256,
        n_steps=64,
        use_tvf=False,
        priority=150,
        policy_mini_batch_size=1024,
        value_mini_batch_size=1024,
        value_epochs=4,
        policy_epochs=4,
        target_kl=1.0,
        ppo_epsilon=0.1,
        value_lr=2.5e-4,
        policy_lr=2.5e-4,
        gamma=0.999,
        default_params=initial_args,
    )

    # this could just be a more realistic target to hit than 0.999?
    # (actually 0.999 is probably fine)
    add_job(
        f"TVF_10_Regression",
        run_name=f"ppo (alt_997)",
        max_grad_norm=20.0,
        agents=256,
        n_steps=64,
        use_tvf=False,
        priority=150,
        policy_mini_batch_size=1024,
        value_mini_batch_size=1024,
        value_epochs=4,
        policy_epochs=4,
        target_kl=1.0,
        ppo_epsilon=0.1,
        value_lr=2.5e-4,
        policy_lr=2.5e-4,
        gamma=0.997,
        default_params=initial_args,
    )

    add_job(
        # the is very close to the old settings
        f"TVF_10_Regression",
        run_name=f"tvf (alt)",
        max_grad_norm=20.0,
        agents=256,
        n_steps=64,
        use_tvf=True,
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,
        priority=150,
        policy_mini_batch_size=1024,
        value_mini_batch_size=1024,
        value_epochs=4,
        policy_epochs=4,
        target_kl=1.0,
        ppo_epsilon=0.1,
        value_lr=2.5e-4,
        policy_lr=2.5e-4,
        default_params=initial_args,
    )

    add_job(
        # the is very close to the old settings
        f"TVF_10_Regression",
        run_name=f"tvf (alt-16)",
        max_grad_norm=20.0,
        agents=256,
        n_steps=64,
        use_tvf=True,
        tvf_lambda=-16, # try lower n-step.. this will be slow... but at least we wont be reusing the one bootstrap estimate...
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,
        priority=150,
        policy_mini_batch_size=1024,
        value_mini_batch_size=1024,
        value_epochs=4,
        policy_epochs=4,
        target_kl=1.0,
        ppo_epsilon=0.1,
        value_lr=2.5e-4,
        policy_lr=2.5e-4,
        default_params=initial_args,
    )

    add_job(
        # try make sure we learn value function fast enough...
        f"TVF_10_Regression",
        run_name=f"tvf (alt-16-hq)",
        max_grad_norm=20.0,
        agents=256,
        n_steps=64,
        use_tvf=True,
        tvf_lambda=-16,
        # try lower n-step.. this will be slow... but at least we wont be reusing the one bootstrap estimate...
        tvf_max_horizon=3000,
        gamma=0.999,
        tvf_gamma=0.999,
        priority=150,
        policy_mini_batch_size=1024,
        value_mini_batch_size=1024,
        value_epochs=2,
        policy_epochs=6,
        target_kl=1.0,
        ppo_epsilon=0.1,
        value_lr=2.5e-4,
        policy_lr=2.5e-4,
        default_params=initial_args,
    )

if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    id = 0
    job_list = []
    setup_experiments9()
    setup_experiments10()

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