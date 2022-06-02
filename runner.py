from runner_tools import *

# ---------------------------------------------------------------------------------------------------------

def print_experiments(job_filter=None):
    cmds = get_experiment_cmds(job_filter)
    for cmd in cmds:
        print(cmd)
        print()

def generate_slurm(job_filter=None):
    """
    Generate slurm scripts for jobs
    """

    template = """#!/bin/bash
#SBATCH --job-name=%JOBNAME%          # Job name
#SBATCH --mail-type=START,END,FAIL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=matthew.aitchison@anu.edu.au     # Where to send mail
#SBATCH --ntasks=8                    # We use 8-workers per job so 16 is ideal, but 8 is ok too.
#SBATCH --mem=16G                     # 8GB per job is about right
#SBATCH --time=4:00:00                # Jobs take 8 hours tops, but maybe do them in chunks?
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080ti:1           # Two jobs per one GPU, 2080ti is fine, but the AMD cores attached to the 3090 are much faster.
#SBATCH --output=%JOBNAME%_%j.log     # Standard output and error log

pwd; hostname; date
echo "--- training ---"
cd PPO     
singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif %CMD1% &
singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif %CMD2% &
wait
echo "--- done ---"
date
"""


    # run with
    # $ sbatch job_001.slurm

    """

    cmds = get_experiment_cmds(job_filter, force_params={'mutex':''})
    n = 0
    while len(cmds) > 0:
        n += 1
        with open(f'job_{n:03d}.slurm', 'wt') as t:
            modified_template = template.replace("%JOBNAME%", f'job_{n:03d}')
            modified_template = modified_template.replace("%CMD1%", cmds.pop(0))
            if len(cmds) > 0:
                modified_template = modified_template.replace("%CMD2%", cmds.pop(0))
            else:
                modified_template = modified_template.replace("%CMD2%", '')

            t.write(modified_template)


if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    # load in the jobs...
    import exp_tvf

    exp_tvf.setup()

    # todo: switch to proper command line args...

    experiment_filter = None

    if len(sys.argv) == 1:
        mode = "show"
    elif len(sys.argv) == 2:
        mode = sys.argv[1]
    elif len(sys.argv) == 3:
        mode = sys.argv[1]
        experiment_filter = sys.argv[2]
    else:
        raise Exception("Invalid parameters.")

    job_filter = (lambda x: experiment_filter in x.run_name) if experiment_filter is not None else None

    if mode == "show_all":
        show_experiments(all=True)
    elif mode == "show":
        show_experiments()
    elif mode == "print":
        print_experiments(job_filter)
    elif mode == "slurm":
        generate_slurm(job_filter)
    elif mode == "clash":
        fix_clashes()
    elif mode == "fps":
        show_fps()
    elif mode == "auto":
        run_next_experiment()
    else:
        run_next_experiment(filter_jobs=lambda x: mode in x.run_name)
