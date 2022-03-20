from runner_tools import *

# ---------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # see https://github.com/pytorch/pytorch/issues/37377 :(
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    # load in the jobs...
    import EXP_RP
    import EXP_TVF
    import EXP_PGG
    import EXP_A57
    import EXP_EXP
    import EXP_DNA
    import EXP_STUCK

    #EXP_TVF.setup(-100)
    EXP_DNA.setup(50)

    if len(sys.argv) == 1:
        experiment_name = "show"
    else:
        experiment_name = sys.argv[1]

    if experiment_name == "show_all":
        show_experiments(all=True)
    elif experiment_name == "show":
        show_experiments()
    elif experiment_name == "clash":
        fix_clashes()
    elif experiment_name == "fps":
        show_fps()
    elif experiment_name == "auto":
        run_next_experiment()
    else:
        run_next_experiment(filter_jobs=lambda x: experiment_name in x.run_name)
