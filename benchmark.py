# run benchmark tests
import os
import sys

def run_benchmark(color, resolution, agents, n_iterations=100):

    # run the test

    os.system(
        ("python ppo.py benchmark"
        " --run_name="+BENCHMARK_NAME+" --experiment_name='color={0} resolution={1} agents={2}'"
        " --color={0} --resolution={1} --agents={2}"
        " --export_video=False"
        #" --device=cuda:1"
        " --n_iterations={3}"
        " --output_folder=/home/matthew/Dropbox/Experiments/ppo").format(color, resolution, agents, n_iterations))

    # extract results
    pass


assert len(sys.argv) == 2
BENCHMARK_NAME = sys.argv[1]

"""
for color in [True, False]:
    for agents in [1,2,4,8,16,32,64,128,256,512,1024]:
        for resolution in ['half', 'standard', 'full']:
            run_benchmark(color, resolution, agents)
"""
run_benchmark(False, "standard", 64, 10)