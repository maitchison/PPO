from tools.runner_tools import *

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


TEMPLATE_GPUSVR = SlurmTemplate("gpusrv", """#!/bin/bash
#SBATCH --job-name=%JOBNAME%          # Job name
#SBATCH --mail-type=END,FAIL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=matthew.aitchison@anu.edu.au     # Where to send mail
#SBATCH --ntasks=24                   # More than 24 seems to crash prolog?
#SBATCH --mem=32G                     # 6GB per job is about right
#SBATCH --time=48:00:00               # Jobs take about 20-hours to run, but can be a bit faster 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3090:2             # Two jobs per one GPU, 2080ti is fine, but the AMD cores attached to the 3090 are much faster.
#SBATCH --output=%j.log     # Standard output and error log

pwd; hostname; date
echo "--- training ---"
cd ~
cd PPO     
%CMD%
echo "--- done ---"
date
""", n_gpus=2, n_jobs=4)


# 2080 machines have 48 cores and 128GB ram

# procgen version...

# mlvc has 48 cores, 128 GB

TEMPLATE_MLVC = SlurmTemplate("mlvc", """#!/bin/bash
#SBATCH --job-name=%JOBNAME%          # Job name
#SBATCH --mail-type=END,FAIL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=matthew.aitchison@anu.edu.au     # Where to send mail
#SBATCH --ntasks=24                   # More than 24 seems to crash prolog?
#SBATCH --mem=30G                     # 6GB per job is about right
#SBATCH --time=48:00:00               # Jobs take about 20-hours to run, but can be a bit faster 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080ti:2           # Two jobs per one GPU, 2080ti is fine, but the AMD cores attached to the 3090 are much faster.
#SBATCH --output=%j.log     # Standard output and error log
pwd; hostname; date
echo "--- training ---"
cd ~
cd PPO     
%CMD%
echo "--- done ---"
date
""", n_gpus=2, n_jobs=4)


def generate_slurm(experiment:str, job_filter=None, st: SlurmTemplate=TEMPLATE_MLVC):
    """
    Generate slurm scripts for jobs
    """

    cluster_filter = lambda job: job.hostname == "cluster" and (job_filter is None or job_filter(job))

    cmds = get_experiment_cmds(
        cluster_filter,
        force_params={
            'mutex_key': '',
            'ignore_lock': True,
            'device': "cuda",
        }
    )
    n = 0
    while len(cmds) > 0:
        n += 1
        with open(f'{experiment}_{n:02d}_{st.name}.slurm', 'wt') as t:

            lines = []

            while len(cmds) > 0 and len(lines) < st.n_jobs:
                cmd = cmds.pop(0)
                cmd = cmd.replace('--device="cuda"', f'--device="cuda:{len(lines) % st.n_gpus}"')
                lines.append(f"singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif {cmd} &")
            lines.append('wait')

            modified_template = st.template.replace("%JOBNAME%", f'{experiment}_{n:02d}').replace("%CMD%", "\n".join(lines))

            t.write(modified_template)


if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    import tvf_exp3
    tvf_exp3.setup()

    # todo: switch to proper command line args...

    experiment_filter = None

    run_filter = None

    if len(sys.argv) == 1:
        mode = "show"
    elif len(sys.argv) == 2:
        mode = sys.argv[1]
    elif len(sys.argv) == 3:
        mode = sys.argv[1]
        experiment_filter = sys.argv[2]
    elif len(sys.argv) == 4: # this is for auto...
        _path = sys.argv[0]
        mode = sys.argv[1]
        _device = sys.argv[2]
        run_filter = lambda x: sys.argv[3] in x.run_name or sys.argv[3] in x.experiment_name
    else:
        raise Exception("Invalid parameters.")

    job_filter = (
        lambda x:
            (experiment_filter.lower() in x.run_name.lower()) or
            (experiment_filter.lower() in x.experiment_name.lower())
    ) if experiment_filter is not None else None

    if mode == "show_all":
        show_experiments(all=True)
    elif mode == "show":
        show_experiments()
    elif mode == "print":
        print_experiments(job_filter)
    elif mode == "slurm":
        print(f"Generating slurm scripts.")
        for template in [TEMPLATE_MLVC, TEMPLATE_GPUSVR, TEMPLATE_MLVC]:
            generate_slurm(experiment=experiment_filter or "job",  job_filter=job_filter, st=template)
        print("Done.")
    elif mode == "clash":
        fix_clashes()
    elif mode == "fps":
        show_fps()
    elif mode == "auto":
        run_next_experiment(filter_jobs=run_filter)
    else:
        run_next_experiment(filter_jobs=lambda x: mode in x.run_name)
