# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Imports all environments so that they register themselves with the Gym API.

This protocol is the same as Minigrid, and allows all environments to be
simultaneously registered with Gym as a package.
"""

# Import all environments and register them, so pylint: disable=wildcard-import
from . import cluttered
from . import coingame
from . import doorkey
from . import empty
from . import fourrooms
from . import gather
from . import lava_walls
from . import maze
from . import meetup
from . import stag_hunt
from . import tag
from . import tasklist