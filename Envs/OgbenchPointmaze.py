import time
from typing import Optional, SupportsFloat, Any, Tuple, Dict, List
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import tqdm  # Make sure to install tqdm: pip install tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ogbench is the new environment source
import ogbench

# A simple module-level cache to store computed ground truth distances
# This prevents re-computation if you create multiple instances of the wrapper for the same env
_GT_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


class OgbenchPointmaze(gym.Wrapper):
    """
    A wrapper for the ogbench, it accepts 'pointmaze-medium-navigate-v0', pointmaze-large-navigate-v0, pointmaze-giant-navigate-v0, pointmaze-medium-stitch-v0, pointmaze-large-stitch-v0, pointmaze-giant-stitch-v0 environments.

    This class provides the standard gym.Wrapper interface and adds a `gt` method
    to compute the ground truth shortest-path distances between all reachable
    states in the maze grid.
    """
    def __init__(self, env_name: str, render_mode: Optional[str] = None):
        # 1. Create the base ogbench environment
        kwargs = {}
        if render_mode is not None:
            kwargs['render_mode'] = render_mode
        env, trainset, testset = ogbench.make_env_and_datasets(env_name, **kwargs)

        self.trainset = trainset
        self.testset = testset
        
        # 2. Initialize the gym.Wrapper with the created environment
        super().__init__(env)
        
        # For convenience, you can store references if you want, but they are also
        # accessible via self.env.observation_space etc.
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # A name for our cache key
        self._env_name = env_name

    # Note: reset and step should be methods of the class, not nested in __init__
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Resets the environment to an initial state and returns the initial observation."""
        return self.env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics."""
        return self.env.step(action)

    def plot_observation(self, obs: np.ndarray, ax=None, title: str = "Maze with Observation", save_path: str = None):
        """
        Plots the maze and the location corresponding to the given observation.
        
        Args:
            obs (np.ndarray): The observation from the environment.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new plot is created.
            title (str, optional): The title for the plot.
            save_path (str, optional): If provided, the plot will be saved to this path.
        """
        show_plot = ax is None
        if show_plot:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        unwrapped_env = self.env.unwrapped
        maze_map = unwrapped_env.maze_map
        
        # Plot the maze (walls are 1, free space is 0)
        ax.imshow(maze_map, cmap='gray')
        
        # Convert observation (x, y) to grid coordinates (i, j)
        xy_pos = obs[:2]
        ij_pos = unwrapped_env.xy_to_ij(xy_pos)
        
        # Plot the agent's position
        # ij_pos is (row, col), but plot wants (x, y) so we use (col, row)
        ax.plot(ij_pos[1], ij_pos[0], 'ro', markersize=10, label='Agent Position')
        
        ax.set_title(title)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        if show_plot and not save_path:
            plt.show()

    def plot_grid_observations(self, title: str = "Grid Cell to Observation State Mapping", save_path: str = None):
        """
        Plots the maze and annotates each free cell with its corresponding observation state.
        
        Args:
            title (str, optional): The title for the plot.
            save_path (str, optional): If provided, the plot will be saved to this path.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        unwrapped_env = self.env.unwrapped
        maze_map = unwrapped_env.maze_map
        
        # Plot the maze
        ax.imshow(maze_map, cmap='gray')
        
        rows, cols = maze_map.shape
        
        # Iterate over all cells to annotate them
        for r in range(rows):
            for c in range(cols):
                if maze_map[r, c] == 0:  # If it's a free cell
                    # Convert grid coordinate (i, j) to observation (x, y)
                    xy_pos = unwrapped_env.ij_to_xy((r, c))
                    
                    # Format the observation for display
                    obs_text = f"({xy_pos[0]:.1f}, {xy_pos[1]:.1f})"
                    
                    # Annotate the cell
                    ax.text(c, r, obs_text, ha='center', va='center', color='yellow', fontsize=6)
        
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def gt(self, max_dist_accuracy: float = None) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Computes and caches the ground truth distances for the pointmaze environment.
        This function is an adaptation of the provided template.
        """
        if self._env_name in _GT_CACHE:
            print(f"Loading ground truth from cache for '{self._env_name}'.")
            # Convert cached numpy arrays to torch tensors for consistency with the other env
            s1, s2, dist = _GT_CACHE[self._env_name]
            return torch.from_numpy(s1).float(), torch.from_numpy(s2).float(), torch.from_numpy(dist).float()

        print(f"Calculating ground truth distances for '{self._env_name}'...")
        
        # Use .unwrapped to access the environment's specific attributes
        unwrapped_env = self.env.unwrapped
        maze_map = unwrapped_env.maze_map
        rows, cols = maze_map.shape

        # Run BFS to get shortest path distances for all pairs of free cells
        ground_truth_grid = self._compute_all_pairs_shortest_path(maze_map)

        # Convert grid coordinates and distances to state vectors
        obs_dim = self.observation_space.shape[0]
        states_s1, states_s2, distances = [], [], []

        print("Converting grid coordinates to state space...")

        state_augmentation = ((-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)) # small shifts to augment states

        for (s1_ij, s2_ij), dist in tqdm.tqdm(ground_truth_grid.items(), desc="Converting to states"):
            # Use the environment's built-in conversion function
            s1_xy = unwrapped_env.ij_to_xy(s1_ij)
            s2_xy = unwrapped_env.ij_to_xy(s2_ij)
            
            # The full state includes velocity, which we assume is zero for GT calculation
            full_s1, full_s2 = np.zeros(obs_dim), np.zeros(obs_dim)
            full_s1[:2], full_s2[:2] = s1_xy, s2_xy
            
            states_s1.append(full_s1)
            states_s2.append(full_s2)
            distances.append(dist)

            augmented_s1, augmented_s2 = full_s1.copy(), full_s2.copy()
            for dx, dy in state_augmentation:
                augmented_s1[0] += dx
                augmented_s1[1] += dy
                augmented_s2[0] += dx
                augmented_s2[1] += dy
                states_s1.append(augmented_s1.copy())
                states_s2.append(augmented_s2.copy())
                distances.append(dist)
        
        # Store the result as numpy arrays in the cache
        result_np = (np.array(states_s1), np.array(states_s2), np.array(distances))
        _GT_CACHE[self._env_name] = result_np
        print("Ground truth calculation complete and cached.")

        # Return as torch tensors to match the other environment's API
        return torch.from_numpy(result_np[0]).float(), torch.from_numpy(result_np[1]).float(), torch.from_numpy(result_np[2]).float()

    def _compute_all_pairs_shortest_path(self, maze_map: np.ndarray) -> Dict[Tuple, int]:
        """Helper function to run BFS from every free cell."""
        rows, cols = maze_map.shape
        
        # Get all valid (free) positions in the grid. maze_map == 0 means free.
        free_cells = [(r, c) for r in range(rows) for c in range(cols) if maze_map[r, c] == 0]

        ground_truth = {}
        for start_ij in tqdm.tqdm(free_cells, desc="Running BFS for GT"):
            dist_map = self._bfs_from_start(start_ij, maze_map)
            for goal_ij, dist in dist_map.items():
                ground_truth[(start_ij, goal_ij)] = dist
        return ground_truth

    def _bfs_from_start(self, start_ij: Tuple[int, int], maze_map: np.ndarray) -> Dict[Tuple, int]:
        """Performs BFS to find the shortest path from a single start_ij."""
        rows, cols = maze_map.shape
        queue = deque([(start_ij, 0)])  # ((i, j), distance)
        visited = {start_ij}
        distances = {}
        
        while queue:
            (i, j), d = queue.popleft()
            distances[(i, j)] = d
            
            # Check neighbors (up, down, left, right)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                
                # Check bounds and if the cell is free and not visited
                if (0 <= ni < rows and 0 <= nj < cols and 
                    maze_map[ni, nj] == 0 and (ni, nj) not in visited):
                    visited.add((ni, nj))
                    queue.append(((ni, nj), d + 1))
        return distances


# Example of how to use the new wrapper
if __name__ == "__main__":
    print("Creating the wrapped Ogbench environment...")
    env_name = 'pointmaze-giant-stitch-v0'  # Change to desired environment name
    env = OgbenchPointmaze(env_name=env_name)

    # --- Test basic environment interaction ---
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step successful. Next observation shape: {next_obs.shape}")
    print("-" * 30)

    # --- Test the new plotting function ---
    print("\nPlotting the agent's initial position...")
    env.plot_observation(obs, title="Initial Agent Position", save_path="initial_position.png")
    print("-" * 30)

    # --- Test the grid observation plotting function ---
    print("\nPlotting the grid to observation mapping...")
    env.plot_grid_observations(save_path=f"{env_name}_grid_observations.png")
    print("-" * 30)

    # --- Test the ground truth function ---
    start_time = time.time()
    states_s1, states_s2, gt_distances = env.gt()
    end_time = time.time()
    
    print(f"\nTime to compute GT: {end_time - start_time:.2f} seconds")
    print(f"Shape of states_s1: {states_s1.shape}")
    print(f"Shape of states_s2: {states_s2.shape}")
    print(f"Shape of gt_distances: {gt_distances.shape}")
    print(f"Found {len(gt_distances)} ground truth state pairs.")

    # Print a sample of the results
    print("\n--- Sample Ground Truth Distances ---")
    for i in range(min(100, len(gt_distances))):
        s1_coord = states_s1[i].numpy().round(2)
        s2_coord = states_s2[i].numpy().round(2)
        dist = gt_distances[i].item()
        print(f"From state {s1_coord} to {s2_coord} -> Distance: {dist}")
        
    print("-" * 30)
    # --- Test the caching mechanism ---
    print("\nCalling env.gt() again to test caching...")
    start_time_cache = time.time()
    _ = env.gt()
    end_time_cache = time.time()
    print(f"Time to load GT from cache: {end_time_cache - start_time_cache:.4f} seconds")