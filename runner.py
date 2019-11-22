import os
import sys
import shutil

experiment_name = sys.argv[1]

def has_run(experiment_folder, run_name):
    """
    Returns if this experiment has been started or not.
    :param output_folder:
    :param experiment_name:
    :param run_name:
    :return:
    """

    if not os.path.exists(experiment_folder):
        return False

    for subdir, dirs, files in os.walk(experiment_folder):
        for file in files:
            name = os.path.split(subdir)[-1]
            if file == "training_log.csv":
                this_run_name = name[:-19] # crop off the id code.
                if this_run_name == run_name:
                    return True

    return False

def run_experiment(experiment_name, env_name, run_name, **kwargs):

    output_folder = "/home/matthew/Dropbox/Experiments/ppo"
    kwargs["output_folder"] = output_folder

    experiment_folder = os.path.join(output_folder, experiment_name)

    # make the destination folder...
    if not os.path.exists(experiment_folder):
        print("Making experiment folder " + experiment_folder)
        os.makedirs(experiment_folder, exist_ok=True)

    # copy script across if needed.
    ppo_path = experiment_folder+ "/ppo.py"
    if not os.path.exists(ppo_path):
        print("Copying ppo.py")
        shutil.copy("ppo.py", ppo_path)

    if has_run(experiment_folder, run_name):
        return

    kwargs["experiment_name"] = experiment_name
    kwargs["run_name"] = run_name

    python_part = "python {} {}".format(ppo_path, env_name)
    params_part = " ".join(["--{}='{}'".format(k,v) for k,v in kwargs.items()])
    params_part_lined = "\n".join(["{}:'{}'".format(k, v) for k, v in kwargs.items()])

    print()
    print("=" * 120)
    print("Running " + python_part + "\n" + params_part_lined)
    print("=" * 120)
    print()
    return_code = os.system(python_part + " " + params_part)
    if return_code != 0:
        raise Exception("Error {}.".format(return_code))


if experiment_name == "GA_Pong":
    # game analysis: pong
    for agents in reversed([8, 16, 32, 64, 128, 256, 512, 1024]):
        run_experiment(
            "GA_Pong", "pong", "agents="+str(agents),
            workers=32, epochs=20,
            agents=agents
        )

if experiment_name == "GA_Pong_RewardClip":
    # game analysis: pong
    for reward_clip in [1, 3, 5]:
        run_experiment(
            "GA_Pong", "pong", "reward_clip="+str(reward_clip),
            workers=32, epochs=20, agents=256,
            reward_clip=reward_clip
        )

if experiment_name == "GA_Pong_MiniBatchSize":
    # game analysis: pong
    for mini_batch_size in [512, 1024, 2048, 4096]:
        run_experiment(
            "GA_Pong", "pong", "mini_batch_size=" + str(mini_batch_size),
            workers=32, epochs=20, agents=256,
            mini_batch_size=mini_batch_size,
        )

elif experiment_name == "RA_Alien":
    # resolution analysis: alien

    """
    for resolution in ["half", "standard", "full"]:
        for model in ["cnn", "cnn_improved"]:
            color = True if model == "improved_cnn" else False
            os.system(
                "python ppo.py alien --run_name=RA_Alien" +
                " --workers=32" +
                " --agents=128" +
                " --experiment_name='resolution={} model={} color={}'".format(resolution, model, color) +
                " --resolution={} --model={} --color={}".format(resolution, model, color) +
                " --output_folder=/home/matthew/Dropbox/Experiments/ppo/"
            )
    """
    pass # redo this...
else:
    raise Exception("Invalid experiment.")