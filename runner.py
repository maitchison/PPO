import os
import sys
import shutil

experiment_name = sys.argv[1]

def run_experiment(env_name, run_name, **kwargs):

    output_folder = "/home/matthew/Dropbox/Experiments/ppo"
    kwargs["output_folder"] = output_folder

    # copy script accross if needed.
    ppo_path = os.path.join(output_folder, run_name )+ "/ppo.py"
    if not os.path.exists(ppo_path):
        shutil.copy("ppo.py", ppo_path)

    kwargs["run_name"] = run_name

    python_part = "python {} {}".format(ppo_path, env_name)
    params_part = " ".join(["{}='{}'".format(k,v) for k,v in kwargs.items()])

    os.system(python_part + " " + params_part)

if experiment_name == "GA_Pong":
    # game analysis: pong
    for agents in reversed([8, 16, 32, 64, 128, 256, 512, 1024]):
        run_experiment(
            "pong", "GA_Pong",
            workers=32, epochs=20,
            agents=agents,
            experiment_name="agents="+str(agents)
        )

if experiment_name == "GA_Pong_RewardClip":
    # game analysis: pong
    for reward_clip in [1, 3, 5]:
        run_experiment(
            "pong", "GA_Pong",
            workers=32, epochs=20, agents=256,
            reward_clip=reward_clip,
            experiment_name="reward_clip="+str(reward_clip)
        )

if experiment_name == "GA_Pong_MiniBatchSize":
    # game analysis: pong
    for mini_batch_size in [512, 1024, 2048, 4096]:
        run_experiment(
            "pong", "GA_Pong",
            workers=32, epochs=20, agents=256,
            mini_batch_size=mini_batch_size,
            experiment_name="mini_batch_size=" + str(mini_batch_size)
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