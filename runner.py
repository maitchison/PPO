from runner_tools import *

# ---------------------------------------------------------------------------------------------------------

def print_experiments(job_filter=None):
    cmds = get_experiment_cmds(job_filter)
    for cmd in cmds:
        print(cmd)
        print()

class SlurmTemplate:
    def __init__(self, name:str, template: str, n_gpus: int, n_jobs:int):
        self.name = name
        self.template = template
        self.n_gpus = n_gpus
        self.n_jobs = n_jobs

TEMPLATE_3090 = SlurmTemplate("3090", """#!/bin/bash
#SBATCH --job-name=%JOBNAME%          # Job name
#SBATCH --mail-type=END,FAIL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=matthew.aitchison@anu.edu.au     # Where to send mail
#SBATCH --ntasks=32                   # We use 8-workers per job so 16 is ideal, but 8 is ok too.
#SBATCH --mem=64G                     # 8GB per job is about right
#SBATCH --time=24:00:00               # Jobs take about 20-hours to run, but can be a bit faster 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3090:2             # Two jobs per one GPU, 2080ti is fine, but the AMD cores attached to the 3090 are much faster.
#SBATCH --output=~/logs/%j.log     # Standard output and error log

pwd; hostname; date
echo "--- training ---"
cd ~
cd PPO     
%CMD%
echo "--- done ---"
date
""", n_gpus=2, n_jobs=8)

TEMPLATE_2080ti = SlurmTemplate("2080ti", """#!/bin/bash
#SBATCH --job-name=%JOBNAME%          # Job name
#SBATCH --mail-type=END,FAIL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=matthew.aitchison@anu.edu.au     # Where to send mail
#SBATCH --ntasks=48                   # We use 8-workers per job so 16 is ideal, but 8 is ok too.
#SBATCH --mem=64G                     # 8GB per job is about right
#SBATCH --time=24:00:00               # Jobs take about 20-hours to run, but can be a bit faster 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080ti:4           # Two jobs per one GPU, 2080ti is fine, but the AMD cores attached to the 3090 are much faster.
#SBATCH --output=~/logs/%j.log     # Standard output and error log
pwd; hostname; date
echo "--- training ---"
cd ~
cd PPO     
%CMD%
echo "--- done ---"
date
""", n_gpus=4, n_jobs=8)

def generate_slurm(job_filter=None, st: SlurmTemplate=TEMPLATE_2080ti):
    """
    Generate slurm scripts for jobs
    """

    cmds = get_experiment_cmds(job_filter, force_params={'mutex_key': ''})
    n = 0
    while len(cmds) > 0:
        n += 1
        with open(f'job_{n:02d}_{st.name}.slurm', 'wt') as t:

            lines = []

            while len(cmds) > 0 and len(lines) < st.n_jobs:
                cmd = cmds.pop(0)
                cmd = cmd.replace('--device="cuda"', f'--device="cuda:{len(lines) % st.n_gpus}"')
                lines.append(f"singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif {cmd} &")
            lines.append('wait')

            modified_template = st.template.replace("%JOBNAME%", f'job_{n:02d}').replace("%CMD%", "\n".join(lines))

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
        generate_slurm(job_filter, TEMPLATE_2080ti)
        generate_slurm(job_filter, TEMPLATE_3090)
    elif mode == "clash":
        fix_clashes()
    elif mode == "fps":
        show_fps()
    elif mode == "auto":
        run_next_experiment()
    else:
        run_next_experiment(filter_jobs=lambda x: mode in x.run_name)
