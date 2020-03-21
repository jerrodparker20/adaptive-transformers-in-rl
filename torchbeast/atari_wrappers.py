# The MIT License
#
# Copyright (c) 2017 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Taken from
#   https://raw.githubusercontent.com/openai/baselines/7c520852d9cf4eaaad326a3d548efc915dc60c10/baselines/common/atari_wrappers.py
# and slightly modified.

import numpy as np
from collections import deque
import deepmind_lab



#########I changed this file to become compatible with Deepmind_Lab API
class createDmLab(object):
  """DeepMind Lab wrapper """

  def __init__(self, level, config, seed,
               runfiles_path=None, level_cache=None):

    self._random_state = np.random.RandomState(seed=seed)
    if runfiles_path:
      deepmind_lab.set_runfiles_path(runfiles_path)
    config = {k: str(v) for k, v in config.items()}
    self._observation_spec = ['RGBD']
    self._env = deepmind_lab.Lab(
        level=level,
        observations=self._observation_spec,
        config=config,
        level_cache=level_cache,
    )

  def _reset(self):
    self._env.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))

  def _observation(self):
    d = self._env.observations()
    return d['RGBD']

  def initial(self):
    self._reset()
    d = self._env.observations()
    return d['RGBD']

  def step(self, action):
    reward = self._env.step(action)#, num_steps=self._num_action_repeats)
    done = np.array(not self._env.is_running())
    if done:
      self._reset()
    observation = np.array(self._observation(),dtype=np.uint8)
    reward = np.array(reward, dtype=np.float32)
    return observation, reward, done

  def close(self):
    self._env.close()

