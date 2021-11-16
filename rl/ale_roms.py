# support for additional roms that match MD5s from ALE

import gym
import atari_py

BELLEMARE_MD5s = """
# from https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/md5.txt
4b27f5397c442d25f0c418ccdacf1926 adventure.bin
35be55426c1fec32dfb503b4f0651572 air_raid.bin
f1a0a23e6464d954e3a9579c4ccd01c8 alien.bin
acb7750b4d0c4bd34969802a7deb2990 amidar.bin
de78b3a064d374390ac0710f95edde92 assault.bin
89a68746eff7f266bbf08de2483abe55 asterix.bin
ccbd36746ed4525821a8083b0d6d2c2c asteroids.bin
826481f6fc53ea47c9f272f7050eedf7 atlantis2.bin
9ad36e699ef6f45d9eb6c4cf90475c9f atlantis.bin
8556b42aa05f94bc29ff39c39b11bff4 backgammon.bin
00ce0bdd43aed84a983bef38fe7f5ee3 bank_heist.bin
819aeeb9a2e11deb54e6de334f843894 basic_math.bin
41f252a66c6301f1e8ab3612c19bc5d4 battle_zone.bin
79ab4123a83dc11d468fb2108ea09e2e beam_rider.bin
136f75c4dd02c29283752b7e5799f978 berzerk.bin
0a981c03204ac2b278ba392674682560 blackjack.bin
c9b7afad3bfd922e006a6bfc1d4f3fe7 bowling.bin
c3ef5c4653212088eda54dc91d787870 boxing.bin
f34f08e5eb96e500e851a80be3277a56 breakout.bin
028024fb8e5e5f18ea586652f9799c96 carnival.bin
b816296311019ab69a21cb9e9e235d12 casino.bin
91c2098e88a6b13f977af8c003e0bca5 centipede.bin
c1cb228470a87beb5f36e90ac745da26 chopper_command.bin
55ef7b65066428367844342ed59f956c crazy_climber.bin
8cd26dcf249456fe4aeb8db42d49df74 crossbow.bin
106855474c69d08c8ffa308d47337269 darkchambers.bin
0f643c34e40e3f1daafd9c524d3ffe64 defender.bin
f0e0addc07971561ab80d9abe1b8d333 demon_attack.bin
36b20c427975760cb9cf4a47e41369e4 donkey_kong.bin
368d88a6c071caba60b4f778615aae94 double_dunk.bin
5aea9974b975a6a844e6df10d2b861c4 earthworld.bin
71f8bacfbdca019113f3f0801849057e elevator_action.bin
94b92a882f6dbaa6993a46e2dcc58402 enduro.bin
6b683be69f92958abe0e2a9945157ad5 entombed.bin
615a3bf251a38eb6638cdc7ffbde5480 et.bin
b8865f05676e64f3bec72b9defdacfa7 fishing_derby.bin
30512e0e83903fc05541d2f6a6a62654 flag_capture.bin
8e0ab801b1705a740b476b7f588c6d16 freeway.bin
081e2c114c9c20b61acf25fc95c71bf4 frogger.bin
4ca73eb959299471788f0b685c3ba0b5 frostbite.bin
211774f4c5739042618be8ff67351177 galaxian.bin
c16c79aad6272baffb8aae9a7fff0864 gopher.bin
8ac18076d01a6b63acf6e2cab4968940 gravitar.bin
f16c709df0a6c52f47ff52b9d95b7d8d hangman.bin
f0a6e99f5875891246c3dbecbf2d2cea haunted_house.bin
fca4a5be1251927027f2c24774a02160 hero.bin
7972e5101fa548b952d852db24ad6060 human_cannonball.bin
a4c08c4994eb9d24fb78be1793e82e26 ice_hockey.bin
e51030251e440cffaab1ac63438b44ae jamesbond.bin
718ae62c70af4e5fd8e932fee216948a journey_escape.bin
5428cdfada281c569c74c7308c7f2c26 kaboom.bin
4326edb70ff20d0ee5ba58fa5cb09d60 kangaroo.bin
6c1f3f2e359dbf55df462ccbcdd2f6bf keystone_kapers.bin
0dd4c69b5f9a7ae96a7a08329496779a king_kong.bin
eed9eaf1a0b6a2b9bc4c8032cb43e3fb klax.bin
534e23210dd1993c828d944c6ac4d9fb koolaid.bin
4baada22435320d185c95b7dd2bcdb24 krull.bin
5b92a93b23523ff16e2789b820e2a4c5 kung_fu_master.bin
8e4cd60d93fcde8065c1a2b972a26377 laser_gates.bin
2d76c5d1aad506442b9e9fb67765e051 lost_luggage.bin
e908611d99890733be31733a979c62d8 mario_bros.bin
df62a658496ac98a3aa4a6ee5719c251 miniature_golf.bin
3347a6dd59049b15a38394aa2dafa585 montezuma_revenge.bin
aa7bb54d2c189a31bb1fa20099e42859 mr_do.bin
87e79cd41ce136fd4f72cc6e2c161bee ms_pacman.bin
36306070f0c90a72461551a7a4f3a209 name_this_game.bin
113cd09c9771ac278544b7e90efe7df2 othello.bin
fc2233fc116faef0d3c31541717ca2db pacman.bin
7e52a95074a66640fcfde124fffd491a phoenix.bin
6d842c96d5a01967be9680080dd5be54 pitfall2.bin
3e90cf23106f2e08b2781e41299de556 pitfall.bin
60e0ea3cbe0913d39803477945e9e5ec pong.bin
4799a40b6e889370b7ee55c17ba65141 pooyan.bin
ef3a4f64b6494ba770862768caf04b86 private_eye.bin
484b0076816a104875e00467d431c2d2 qbert.bin
393948436d1f4cc3192410bb918f9724 riverraid.bin
ce5cc62608be2cd3ed8abd844efb8919 road_runner.bin
4f618c2429138e0280969193ed6c107e robotank.bin
240bfbac5163af4df5ae713985386f92 seaquest.bin
dd0cbe5351551a538414fb9e37fc56e8 sir_lancelot.bin
b76fbadc8ffb1f83e2ca08b6fb4d6c9f skiing.bin
e72eb8d4410152bdcb69e7fba327b420 solaris.bin
72ffbef6504b75e69ee1045af9075f66 space_invaders.bin
b702641d698c60bcdc922dbd8c9dd49c space_war.bin
a3c1c70024d7aabb41381adbfb6d3b25 star_gunner.bin
a9531c763077464307086ec9a1fd057d superman.bin
4d7517ae69f95cfbc053be01312b7dba surround.bin
42cdd6a9e42a3639e190722b8ea3fc51 tennis.bin
b0e1ee07fbc73493eac5651a52f90f00 tetris.bin
0db4f4150fecf77e4ce72ca4d04c052f tic_tac_toe_3d.bin
fc2104dd2dadf9a6176c1c1c8f87ced9 time_pilot.bin
fb27afe896e7c928089307b32e5642ee trondead.bin
7a5463545dfb2dcfdafa6074b2f2c15e turmoil.bin
085322bae40d904f53bdcc56df0593fc tutankham.bin
a499d720e7ee35c62424de882a3351b6 up_n_down.bin
3e899eba0ca8cd2972da1ae5479b4f0d venture.bin
539d26b6e9df0da8e7465f0f5ad863b7 video_checkers.bin
f0b7db930ca0e548c41a97160b9f6275 videochess.bin
3f540a30fdee0b20aed7288e4a5ea528 videocube.bin
107cc025334211e6d29da0b6be46aec7 video_pinball.bin
7e8aa18bc9502eb57daaf5e7c1e94da7 wizard_of_wor.bin
ec3beb6d8b5689e867bafb5d5f507491 word_zapper.bin
c5930d0e8cdae3e037349bfa08e871be yars_revenge.bin
eea0da9b987d661264cce69a7c13c3bd zaxxon.bin
"""

def install_roms(force=False):
    # step 1, download roms if needed using curl
    # step 2, unpack roms
    # step 3, look through roms and find ones matching MD5s
    # step 4, copy roms to correct path
    #atari_py.get_game_path()
    pass


# additional games to register...
for game in ['ale_skiing']:
    for obs_type in ['image', 'ram']:
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False

        gym.register(
            id='{}-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},
            max_episode_steps=10000,
            nondeterministic=nondeterministic,
        )

        gym.register(
            id='{}-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        # Standard Deterministic (as in the original DeepMind paper)
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # Use a deterministic frame skip.
        gym.register(
            id='{}Deterministic-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip, 'repeat_action_probability': 0.25},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        gym.register(
            id='{}Deterministic-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        gym.register(
            id='{}NoFrameskip-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1, 'repeat_action_probability': 0.25}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

        # No frameskip. (Atari has no entropy source, so these are
        # deterministic environments.)
        gym.register(
            id='{}NoFrameskip-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )
