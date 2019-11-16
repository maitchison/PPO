# run benchmark tests
import os

def run_benchmark(color, resolution, agents):

    # run the test

    os.system(
        ("python ppo.py benchmark"
        " --run_name=benchmark --experiment_name='color={0} resolution={1} agents={2}'"
        " --color={0} --resolution={1} --agents={2}"
        " --export_video=False  --profile"
        " --output_folder=/home/matthew/Dropbox/Experiments/ppo").format(color, resolution, agents))

    # extract results
    pass


for color in [True, False]:
    for agents in [1,2,4,8,16,32,64,128,256,512,1024]:
        for resolution in ['half', 'standard', 'full']:
            run_benchmark(color, resolution, agents)