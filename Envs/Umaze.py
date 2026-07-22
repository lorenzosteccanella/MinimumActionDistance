import time
from typing import Optional, SupportsFloat, Any, Tuple

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from collections import deque

import torch

U_MAZE = [[1, 1, 1, 1, 1],
          [1, 0, 0, 0, 1],
          [1, 1, 1, 0, 1],
          [1, 0, 0, 0, 1],
          [1, 1, 1, 1, 1]]


class Umaze(gym.Wrapper):
    """
    Wrapper for PointMaze_UMaze-v3 that provides a ground truth distance proxy.
    """

    def __init__(self):
        gym.register_envs(gymnasium_robotics)
        super().__init__(gym.make('PointMaze_UMaze-v3'))
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),  # (x, y, vel_x, vel_y)
            dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment and return normalized state."""
        state, info = self.env.reset(seed=seed, options=options)
        return state["observation"], info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, dict[str, Any]]:
        """Take action and return normalized next state."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return next_state["observation"], reward, terminated, truncated, info

    def state_to_grid(self, x, y):
        """Transforms continuous coordinates to grid coordinates."""
        y_prime = int(round(x))
        x_prime = int(round(y))
        y = 3 if y_prime == 1 else (1 if y_prime == -1 else 2)
        x = 1 if x_prime == 1 else (3 if x_prime == -1 else 2)
        return x, y

    def grid_to_state(self, x, y):
        """Transforms grid coordinates to continuous representation."""
        y_prime = 1 if y == 3 else (-1 if y == 1 else 0)
        x_prime = 1 if x == 1 else (-1 if x == 3 else 0)
        return y_prime, x_prime

    def gt(self, max_dist_accuracy=None):
        """Computes shortest path distances using BFS for all valid state pairs."""
        print("Calculating ground truth distances...")
        rows, cols = len(U_MAZE), len(U_MAZE[0])

        # Get all valid (free) positions in the grid
        free_cells = [(x, y) for x in range(rows) for y in range(cols) if U_MAZE[x][y] == 0]

        # Compute shortest paths between all free positions
        ground_truth = {}
        for start in free_cells:
            dist_map = self.bfs_shortest_path(start)
            for goal, dist in dist_map.items():
                ground_truth[(start, goal)] = dist

        # Convert to required output format
        states_s1, states_s2, gt = [], [], []
        for (s1, s2), dist in ground_truth.items():
            transformed_s1 = self.grid_to_state(*s1)
            transformed_s2 = self.grid_to_state(*s2)
            # add (0, 0) for the vel x and vel y
            transformed_s1 = np.concatenate([transformed_s1, [0, 0]])
            transformed_s2 = np.concatenate([transformed_s2, [0, 0]])
            states_s1.append(transformed_s1)
            states_s2.append(transformed_s2)
            gt.append(dist)

        return torch.FloatTensor(states_s1), torch.FloatTensor(states_s2), torch.FloatTensor(gt)

    def bfs_shortest_path(self, start):
        """Performs BFS to find the shortest path from `start` to all reachable cells."""
        queue = deque([(start, 0)])  # (cell, distance)
        visited = set()
        distances = {}

        while queue:
            (x, y), d = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            distances[(x, y)] = d

            # Possible movement directions (up, down, left, right)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(U_MAZE) and 0 <= ny < len(U_MAZE[0]) and U_MAZE[nx][ny] == 0:
                    queue.append(((nx, ny), d + 1))

        return distances


if __name__ == '__main__':
    env = Umaze()

    # Compute ground truth distances
    states_s1, states_s2, gt = env.gt()

    # Print a sample of the results
    for i in range(20):  # Print the first 10 distances
        print(f"From {states_s1[i]} to {states_s2[i]} -> Distance: {gt[i]}")
