# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

from gym.envs.registration import register

register(
    id='atari-v0',
    entry_point='src.envs:AtariEnv',
)