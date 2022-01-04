from runner_tools import WORKERS, add_job, random_search, Categorical
from runner_tools import PPO_reference_args, DNA_reference_args, TVF_reference_args

ROLLOUT_SIZE = 128*128
ATARI_5 = ['Centipede', 'CrazyClimber', 'Krull', 'SpaceInvaders', 'Zaxxon']  # Atari5
