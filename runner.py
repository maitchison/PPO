import os
import sys
import shutil

OUTPUT_FOLDER = "/home/matthew/Dropbox/Experiments/ppo"

def get_status(experiment_folder, run_name):
    """
    Returns if this experiment has been started or not.
    :param output_folder:
    :param experiment_name:
    :param run_name:
    :return:
    """

    if not os.path.exists(experiment_folder):
        return "pending"

    status = "pending"

    for subdir, dirs, files in os.walk(experiment_folder):

        for file in files:
            name = os.path.split(subdir)[-1]
            this_run_name = name[:-19]  # crop off the id code.
            if this_run_name != run_name:
                continue

            # look for the params file (indicates job has stared...
            if file == "params.txt" and status == "pending":
                status = "started"

            # look for the completed file.
            if file == "final_score.txt":
                status = "completed"

    return status

def add_job(experiment_name, run_name, priority=0, **kwargs):
    global id
    job_list.append((priority, id, experiment_name, run_name, kwargs))
    id += 1

def run_experiment(experiment_name, run_name, **kwargs):

    kwargs["output_folder"] = OUTPUT_FOLDER

    experiment_folder = os.path.join(OUTPUT_FOLDER, experiment_name)

    # make the destination folder...
    if not os.path.exists(experiment_folder):
        print("Making experiment folder " + experiment_folder)
        os.makedirs(experiment_folder, exist_ok=True)

    # copy script across if needed.
    ppo_path = experiment_folder+ "/ppo.py"
    if not os.path.exists(ppo_path):
        print("Copying ppo.py")
        shutil.copy("ppo.py", ppo_path)

    kwargs["experiment_name"] = experiment_name
    kwargs["run_name"] = run_name

    python_part = "python {} {}".format(ppo_path, kwargs["env_name"])

    params_part = " ".join(["--{}='{}'".format(k,v) for k,v in kwargs.items() if k not in ["env_name"]])
    params_part_lined = "\n".join(["{}:'{}'".format(k, v) for k, v in kwargs.items()])

    print()
    print("=" * 120)
    print("Running " + python_part + "\n" + params_part_lined)
    print("=" * 120)
    print()
    return_code = os.system(python_part + " " + params_part)
    if return_code != 0:
        raise Exception("Error {}.".format(return_code))

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

    for reward_clip in [1, 3, 5]:
        add_job(
            "GA_Pong",
            run_name="reward_clip=" + str(reward_clip),
            env_name="pong",
            epochs=20,
            agents=256,
            reward_clip=reward_clip
        )

    for mini_batch_size in [256, 512, 1024, 2048, 4096]:
        add_job(
            "GA_Pong",
            run_name="mini_batch_size="+str(mini_batch_size),
            env_name="pong",
            epochs=20,
            agents=256,
            mini_batch_size=mini_batch_size,
        )

    for mini_batch_size in [256, 512, 1024, 2048, 4096]:
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
    # Hash
    # -------------------------------------------------------------------------------------------

    for hash_size in [1, 2, 4, 8, 16, 32]:
        add_job(
            "hash",
            run_name="hash_size=" + str(hash_size),
            env_name="pong",
            epochs=20,
            agents=64,
            filter="hash",
            hash_size=hash_size
        )

    # try the best one for longer...
    add_job(
        "hash",
        run_name="hash_size=8 epochs=200",
        env_name="pong",
        epochs=200,
        agents=64,
        filter="hash",
        hash_size=8,
    )

    # try the best one for longer...
    add_job(
        "hash",
        run_name="hash_size=6 epochs=200 lr=1e-4",
        env_name="pong",
        epochs=200,
        agents=64,
        learning_rate=1e-4,
        filter="hash",
        hash_size=8,
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

def run_next_experiment(filter_experiments = None):
    job_list.sort(key=lambda x: (-x[0], x[1])) # just sort by priority and insertion order.
    for priority, id, experiment_name, run_name, params in job_list:
        if filter_experiments is not None:
            if filter_experiments.lower() != experiment_name.lower():
                continue
        status = get_status(os.path.join(OUTPUT_FOLDER, experiment_name), run_name)
        if status == "pending":
            run_experiment(experiment_name, run_name, **params)
            return


def show_experiments(filter_experiments=None, all=False):
    job_list.sort(key=lambda x: (-x[0], x[1])) # just sort by priority and insertion order.
    print("-" * 100)
    print("{:<10}{:<20}{:60}{:10}".format("priority", "experiment_name", "run_name", "status"))
    print("-" * 100)
    for priority, id, experiment_name, run_name, params in job_list:
        if filter_experiments is not None:
            if filter_experiments.lower() != experiment_name.lower():
                continue
        status = get_status(os.path.join(OUTPUT_FOLDER, experiment_name), run_name)

        if status == "completed" and not all:
            continue

        status_transform = {
            "missing": "",
            "pending": "",
            "completed": "[done]",
            "started": "..."
        }

        print("{:<10}{:<20}{:<60}{:^10}".format(priority, experiment_name, run_name, status_transform[status]))

if __name__ == "__main__":
    id = 0
    job_list = []
    setup_jobs()
    experiment_name = sys.argv[1]
    if experiment_name == "show_all":
        show_experiments(all=True)
    elif experiment_name == "show":
        show_experiments()
    elif experiment_name == "auto":
        run_next_experiment()
    else:
        run_next_experiment(filter_experiments = experiment_name)