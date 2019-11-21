import os
import sys

experiment_name = sys.argv[1]

if experiment_name == "GA_Pong":
    # game analysis: pong
    for agents in reversed([8, 16, 32, 64, 128, 256, 512, 1024]):
        os.system(
            "python ppo.py pong --run_name=GA_Pong"+
            " --workers=32" +
            " --epochs=20" +
            " --experiment_name=agents={0:} --agents={0:}".format(agents) +
            " --output_folder=/home/matthew/Dropbox/Experiments/ppo/"
        )
elif experiment_name == "RA_Alien":
    # resolution analysis: alien
    for resolution in ["half", "standard", "full"]:
        for model in ["cnn", "cnn_improved"]:
            color = True if model == "cnn_improved" else False
            os.system(
                "python ppo.py alien --run_name=RA_Alien"+
                " --workers=32" +
                " --agents=128" +
                " --experiment_name='resolution={} model={} color={}'".format(resolution, model, color) +
                " --resolution={} --model={} --color={}'".format(resolution, model, color) +
                " --output_folder=/home/matthew/Dropbox/Experiments/ppo/"
            )
else:
    raise Exception("Invalid experiment.")