"""Multi-agent goal-seeking task with many static obstacles.
"""
import time
import gym_minigrid.minigrid as minigrid

from envs.gym_multigrid import multigrid
from envs.gym_multigrid.register import register

import random
import math
import numpy as np

class ClutteredMultiGrid(multigrid.MultiGridEnv):
  """Goal seeking environment with obstacles."""

  def __init__(self, size=15, n_agents=3, n_clutter=25, randomize_goal=True,
               agent_view_size=5, max_steps=250, walls_are_lava=False,
               **kwargs):
    self.n_clutter = n_clutter
    self.randomize_goal = randomize_goal
    self.walls_are_lava = walls_are_lava
    super().__init__(grid_size=size, max_steps=max_steps, n_agents=n_agents,
                     agent_view_size=agent_view_size, **kwargs)

  def _gen_grid(self, width, height):
    self.grid = multigrid.Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    if self.randomize_goal:
      self.place_obj(minigrid.Goal(), max_tries=100)
    else:
      self.put_obj(minigrid.Goal(), width - 2, height - 2)
    for _ in range(self.n_clutter):
      if self.walls_are_lava:
        self.place_obj(minigrid.Lava(), max_tries=100)
      else:
        self.place_obj(minigrid.Wall(), max_tries=100)

    self.place_agent()

    self.mission = 'get to the green square'

  def step(self, action):
    obs, reward, done, info = multigrid.MultiGridEnv.step(self, action)
    return obs, reward, done, info


class ClutteredMultiGridFixed15x15(ClutteredMultiGrid):
  """A cluttered environment where the walls and goal are fixed, but agents positions change."""

  def __init__(self, **kwargs):
    super().__init__(n_agents=3, size=15, n_clutter=30, randomize_goal=False,
                     agent_view_size=5, max_steps=100, fixed_environment=True, **kwargs)

  
  def place_agent(self, top=None, size=None, rand_dir=True, max_tries=10000):
    """Override base class to ensure agents' positions change."""

    # Set the random seed to something new so that the positions are different
    self.seed(round(time.time() * 1000))

    for a in range(self.n_agents):
      self.place_one_agent(
          a, top=top, size=size, rand_dir=rand_dir, max_tries=max_tries)


class AdaptiveClutteredMultiGridFixed15x15(ClutteredMultiGrid):
    """A cluttered environment where the walls fixed, but agents positions
    change every block_move_frequency episodes. There are multiple goals
    which move every goal_move_frequency episodes."""

    def __init__(self, block_move_frequency=10, goal_move_frequency=10, **kwargs):
      self.block_move_frequency = block_move_frequency  # Change this as needed
      self.goal_move_frequency = goal_move_frequency # Change this as needed
      self.episode_count = 0
      self.wall_locs = np.array([], dtype=int).reshape(0, 2)  # Empty 2D array for wall locations
      self.goal_locs = np.array([], dtype=int).reshape(0, 2)  # Empty 2D array for goal locations
      super().__init__(n_agents=3, size=15, n_clutter=30, randomize_goal=False,
                      agent_view_size=5, max_steps=5, fixed_environment=True, **kwargs)

    def reset(self):
        """Override parent class reset() to not persist agent locations across episodes when grid is reset.
        Grid is not reset() every time unlike parent class."""
        for i in range(self.n_agents):
          if self.episode_count > 0:
              pos = self.agent_pos[i]
              self.grid.set(pos[0], pos[1], None) # Empty the cell
        return super().reset()

    def _gen_grid(self, width, height):
        """Override parent class reset() method to move a block every block_move_frequency episodes."""
        if self.episode_count == 0:
          # Create a random grid on the first episode
          self.grid = multigrid.Grid(width, height)
          self.grid.wall_rect(0, 0, width, height)

          # Place the goals (n_agents # of goals) in random initial location
          # and store their locations
          for _ in range(self.n_agents):
            goal_loc = self.place_obj(minigrid.Goal(), max_tries=100)
            self.goal_locs = np.append(self.goal_locs, [goal_loc], axis=0)
          
          # Place walls and store their locations
          for _ in range(self.n_clutter):
            wall_loc = self.place_obj(minigrid.Wall(), max_tries=100)
            self.wall_locs = np.append(self.wall_locs, [wall_loc], axis=0)

        for i in range(self.n_agents):
          self.place_one_agent(i)

        if self.episode_count % self.block_move_frequency == 0:
            self.move_random_block()

        if self.goal_move_frequency is None:
          # Goal does not move
          pass
        elif (self.episode_count > 0) and (self.episode_count % self.goal_move_frequency == 0):
          # Move each goal
          for goal_idx in range(self.n_agents):
            self.move_goal(goal_idx)

        self.mission = 'get to the green square'

        self.episode_count += 1

    def move_random_block(self):
        """Move a random block to a random empty cell."""
        wall_idx = random.randint(0, len(self.wall_locs) - 1)
        remove_wall_loc = self.wall_locs[wall_idx]
        self.grid.set(remove_wall_loc[0], remove_wall_loc[1], None) # Empty the cell
        new_wall_loc = self.place_obj(minigrid.Wall(), max_tries=100)
        self.wall_locs[wall_idx] = new_wall_loc

    def move_goal(self, goal_idx):
        """Move each goal by 1 square to a randomly-chosen
        adjacent index."""
        remove_goal_loc = np.copy(self.goal_locs[goal_idx]) # copy to avoid pointer issue
        self.grid.set(remove_goal_loc[0], remove_goal_loc[1], None) # Empty the cell

        # Pick a random direction to move the goal.
        # Possible moves: up, down, left, right
        possible_moves = np.array([
            [remove_goal_loc[0] - 1, remove_goal_loc[1]], # up
            [remove_goal_loc[0] + 1, remove_goal_loc[1]], # down
            [remove_goal_loc[0], remove_goal_loc[1] - 1], # left
            [remove_goal_loc[0], remove_goal_loc[1] + 1]  # right
        ])
        
        # Filter out moves that are out of bounds or occupied
        valid_moves = np.array([
            [x, y] for x, y in possible_moves
            if 0 <= x < self.grid.width and 0 <= y < self.grid.height and self.grid.get(x, y) is None
        ])
        
        if valid_moves.size > 0: # If a valid move exists
            new_goal_loc = random.choice(valid_moves)
            self.grid.set(new_goal_loc[0], new_goal_loc[1], minigrid.Goal())
            self.goal_locs[goal_idx] = new_goal_loc

        if (self.goal_locs[goal_idx] == remove_goal_loc).all():
            # If the goal did not move, notify
            print("Goal did not move. New location is same as old location.")

    def place_one_agent(self,
                      agent_id,
                      top=None,
                      size=None,
                      rand_dir=True,
                      max_tries=math.inf,
                      agent_obj=None):
      """Override this method to place agents at fixed positions."""

      self.agent_pos[agent_id] = None

      if agent_id == 0:
        self.agent_pos[agent_id] = np.array([1, 1])
      elif agent_id == 1:
        self.agent_pos[agent_id] = np.array([1, self.grid.height - 2])
      elif agent_id == 2:
        self.agent_pos[agent_id] = np.array([self.grid.width - 2, 1])

      self.place_agent_at_pos(agent_id, self.agent_pos[agent_id], 
                              agent_obj=agent_obj, rand_dir=rand_dir)

      return self.agent_pos[agent_id]


class ClutteredMultiGridSingle6x6(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, size=6, n_clutter=5, randomize_goal=True,
                     agent_view_size=5, max_steps=50, **kwargs)


class ClutteredMultiGridSingle(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, size=15, n_clutter=25, randomize_goal=True,
                     agent_view_size=5, max_steps=250, **kwargs)


class Cluttered40Minigrid(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=40, minigrid_mode=True, **kwargs)


class Cluttered10Minigrid(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=10, minigrid_mode=True, **kwargs)


class Cluttered50Minigrid(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=50, minigrid_mode=True, **kwargs)


class Cluttered5Minigrid(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=5, minigrid_mode=True, **kwargs)


class Cluttered1MinigridMini(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=1, minigrid_mode=True, size=6,
                     **kwargs)


class Cluttered6MinigridMini(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=6, minigrid_mode=True, size=6,
                     **kwargs)


class Cluttered7MinigridMini(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=7, minigrid_mode=True, size=6,
                     **kwargs)


class ClutteredMinigridLava(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, walls_are_lava=True, minigrid_mode=True,
                     **kwargs)


class ClutteredMinigridLavaMini(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=4, walls_are_lava=True, size=6,
                     minigrid_mode=True, **kwargs)


class ClutteredMinigridLavaMedium(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=15, walls_are_lava=True, size=10,
                     minigrid_mode=True, **kwargs)


class Cluttered15MinigridMedium(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=15, minigrid_mode=True, size=10,
                     **kwargs)

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-Cluttered-v0',
    entry_point=module_path + ':ClutteredMultiGrid'
)

register(
    env_id='MultiGrid-Cluttered-Fixed-15x15',
    entry_point=module_path + ':ClutteredMultiGridFixed15x15'
)

# For new multigrid env
register(
    env_id='MultiGrid-AdaptiveCluttered-Fixed-15x15',
    entry_point=module_path + ':AdaptiveClutteredMultiGridFixed15x15'
)

register(
    env_id='MultiGrid-Cluttered-Single-v0',
    entry_point=module_path + ':ClutteredMultiGridSingle'
)

register(
    env_id='MultiGrid-Cluttered-Single-6x6-v0',
    entry_point=module_path + ':ClutteredMultiGridSingle6x6'
)

register(
    env_id='MultiGrid-Cluttered40-Minigrid-v0',
    entry_point=module_path + ':Cluttered40Minigrid'
)

register(
    env_id='MultiGrid-Cluttered10-Minigrid-v0',
    entry_point=module_path + ':Cluttered10Minigrid'
)

register(
    env_id='MultiGrid-Cluttered50-Minigrid-v0',
    entry_point=module_path + ':Cluttered50Minigrid'
)

register(
    env_id='MultiGrid-Cluttered5-Minigrid-v0',
    entry_point=module_path + ':Cluttered5Minigrid'
)

register(
    env_id='MultiGrid-MiniCluttered1-Minigrid-v0',
    entry_point=module_path + ':Cluttered1MinigridMini'
)

register(
    env_id='MultiGrid-MiniCluttered6-Minigrid-v0',
    entry_point=module_path + ':Cluttered6MinigridMini'
)

register(
    env_id='MultiGrid-MiniCluttered7-Minigrid-v0',
    entry_point=module_path + ':Cluttered7MinigridMini'
)

register(
    env_id='MultiGrid-Cluttered-Lava-Minigrid-v0',
    entry_point=module_path + ':ClutteredMinigridLava'
)

register(
    env_id='MultiGrid-MiniCluttered-Lava-Minigrid-v0',
    entry_point=module_path + ':ClutteredMinigridLavaMini'
)

register(
    env_id='MultiGrid-MediumCluttered-Lava-Minigrid-v0',
    entry_point=module_path + ':ClutteredMinigridLavaMedium'
)

register(
    env_id='MultiGrid-MediumCluttered15-Minigrid-v0',
    entry_point=module_path + ':Cluttered15MinigridMedium'
)

