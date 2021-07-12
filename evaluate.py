"""
Run evaluations on a folder.
"""

import ast
import time
import argparse
import os
import sys
from pathlib import Path

from shutil import copyfile

DEVICE = "cuda:1"
REWARD_SCALE = float()
SAMPLES = 100 # 16 is too few, might need 256...
PARALLEL_ENVS = 64 # number of environments to run in parallel

GENERATE_EVAL = False
GENERATE_MOVIES = True
ZERO_TIME = False

def update_file(source, destination):
    # check if file needs updating
    if not os.path.exists(destination) or os.path.getmtime(destination) != os.path.getmtime(source):
        copyfile(source, destination)
        print(f"<updated {destination}>")

def run_evaluation_script(checkpoint: str, output_file:str, mode:str, temperature=None, samples=None, seed=None):
    """
    Runs "run_evaluation.py" on given checkpoint to produce evaluation / video results.
    """

    *context, experiment_folder, run_name, checkpoint = Path(checkpoint).parts

    experiment_folder = os.path.join(*context, experiment_folder)

    # copy file eval script
    update_file("./run_evaluation.py", os.path.join(experiment_folder, 'run_evaluation.py'))

    args = []

    args.append(mode)
    args.append(os.path.join(run_name, checkpoint))
    args.append(output_file)

    if temperature is not None:
        args.append('--temperature')
        args.append(str(temperature))
    if samples is not None:
        args.append('--samples')
        args.append(str(samples))
    if seed is not None:
        args.append('--seed')
        args.append(str(seed))

    old_path = os.getcwd()
    try:
        os.chdir(experiment_folder)
        #print(" ".join(["python", "run_evaluation.py", *args]))
        import subprocess
        completed_process = subprocess.run(["python", "run_evaluation.py", *args])
        if completed_process.returncode >= 128:
            # fatal error
            sys.exit(completed_process.returncode)
    finally:
        os.chdir(old_path)

def evaluate_run(run_path, temperature, max_epoch:int = 200, seed=None):
    """
    Evaluate all runs in run_path, will skip already done evaluations
    """
    base, folder = os.path.split(run_path)
    if folder in ["rl", "__pycache__"]:
        return

    files_in_dir = [os.path.join(run_path, x) for x in os.listdir(run_path)]

    for epoch in range(0, max_epoch+1):

        postfix = f"_t={temperature}" if temperature is not None else ""

        if ZERO_TIME:
            postfix = postfix + "_no_time"

        if seed is not None:
            postfix = postfix + f"_{seed}"

        checkpoint_name = os.path.join(run_path, f"checkpoint-{epoch:03d}M-params.pt")
        checkpoint_movie_base = f"checkpoint-{epoch:03d}M-eval{postfix}"
        checkpoint_eval_file = os.path.join(os.path.split(run_path)[-1], f"checkpoint-{epoch:03d}M-eval{postfix}.dat")
        checkpoint_full_path_eval_file = os.path.join(run_path, f"checkpoint-{epoch:03d}M-eval{postfix}.dat")

        if os.path.exists(checkpoint_name):

            if GENERATE_MOVIES:

                matching_files = [x for x in files_in_dir if (checkpoint_movie_base in x) and (x.endswith('mp4'))]

                if len(matching_files) >= 2:
                    print(f"Multiple matches for file {run_path}/{checkpoint_movie_base}.")
                    continue

                if len(matching_files) == 1:
                    last_modifed = os.path.getmtime(matching_files[0])
                    file_size = os.path.getsize(matching_files[0])
                    minutes_since_modified = (time.time()-last_modifed)/60
                    needs_redo = file_size < 1024 and minutes_since_modified > 30
                    if needs_redo:
                        print(f" - found stale video {matching_files[0]}, regenerating.")
                        os.remove(matching_files[0])
                else:
                    needs_redo = False

                if len(matching_files) == 0 or needs_redo:
                    output_file = os.path.join(os.path.split(run_path)[-1], checkpoint_movie_base+".mp4")

                    run_evaluation_script(
                        mode='video_nt' if ZERO_TIME else 'video',
                        checkpoint=checkpoint_name,
                        output_file=output_file,
                        temperature=temperature
                    )

            if GENERATE_EVAL and not os.path.exists(checkpoint_full_path_eval_file):
                run_evaluation_script(
                    mode='eval',
                    checkpoint=checkpoint_name,
                    output_file=checkpoint_eval_file,
                    temperature=temperature,
                    seed=seed,
                    samples=SAMPLES
                )


def monitor(path, experiment_filter=None, max_epoch=200, seed=None):

    folders = [x[0] for x in os.walk(path)]
    for folder in folders:
        if experiment_filter is not None and not experiment_filter(folder):
            continue
        if os.path.split(folder)[-1] == "rl":
            continue
        try:
            for temperature in temperatures:
                evaluate_run(folder, temperature=temperature, max_epoch=max_epoch, seed=seed)
        except Exception as e:
            print("Error:"+str(e))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generates videos and evaluation results")
    parser.add_argument("mode", help="[video|eval]")
    parser.add_argument("run_filter", type=str, default="", help="Filter for runs.")
    parser.add_argument("--experiment_filter", type=str, default="", help="Filter for experiments.")
    parser.add_argument("--temperatures", type=str, default="[None]",
                        help="Temperatures to generate. e.g. [0.1, 0.5, 1.0] (lower temperature has higher entropy)")
    parser.add_argument("--max_epoch", type=int, default=200, help="Max number of epochs to test up to.")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed to use.")
    eval_args = parser.parse_args()

    assert eval_args.mode in ["video", "video_nt", "eval"]

    ZERO_TIME = eval_args.mode == "video_nt"
    GENERATE_EVAL = eval_args.mode == "eval"
    GENERATE_MOVIES = eval_args.mode == "video"

    temperatures = ast.literal_eval(eval_args.temperatures)
    if type(temperatures) in [float, int]:
        temperatures = [temperatures]

    folders = [name for name in os.listdir("./Run/") if os.path.isdir(os.path.join('./Run', name))]

    for max_epoch in range(0, 201, 5):
        for folder in folders:
            if eval_args.run_filter in folder:
                if eval_args.experiment_filter:
                    exp_filter = lambda x : eval_args.experiment_filter in x
                else:
                    exp_filter = None
                monitor(
                    os.path.join('./Run', folder),
                    experiment_filter=exp_filter,
                    max_epoch=max_epoch,
                    seed=eval_args.seed
                )