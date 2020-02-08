""" This script tests the effectiveness of no-op starts in terms of the number of different states they generate.

    We test this by performing between 0 and 30 no-ops at the begining of an episode, then applying a sequence of
    randomly generated actions until frame 60. After this we check if the last 4 frames are identical or not.
"""

import numpy as np
import gym
import random
from collections import defaultdict

atari_games = ['adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']

# get a list of all atari games

random_actions = np.random.randint(0,999,100)

final_states = defaultdict(list)

for game in atari_games:
    name = ''.join([g.capitalize() for g in game.split('_')])
    print("Testing "+name)
    env = gym.make(name+"NoFrameskip-v4")
    action_count = env.action_space.n

    for i in range(30):

        # todo, get the actual no-op action
        env.reset()
        for j in range(30):
            if j < i:
                env.step(0)
            else:
                env.step(random_actions[j] % action_count )

        # wait 30 frames and see what happens
        for j in range(30):
            env.step(random_actions[30+j] % action_count)

        # save the next 4 frames
        state = []
        for j in range(4):
            obs, _, _, _ = env.step(0)
            state.append(obs)

        state = np.concatenate(state, axis=0)
        final_states[game].append(state)


