import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, SupportsFloat

from numpy import ndarray


class NormalizedCliffWalking(gym.Wrapper):
    """
    Wrapper for CliffWalking-v0 that normalizes the state space to (x,y) coordinates.
    """

    def __init__(self, p: float = 1.0):
        """
        Initialize the normalized CliffWalking environment.

        Args:
            p (float): Probability of action success (kept for consistency with other envs)
        """
        super().__init__(gym.make("CliffWalking-v1"))
        self.p = p
        self.height = 4  # Corrected from 3 to actual height of 4
        self.width = 12  # Corrected from 11 to actual width of 12
        self.steps=0

        # Override observation space for normalized (x,y) coordinates
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),  # (x, y) coordinates
            dtype=np.float32
        )

    def _normalize_state(self, state: int) -> np.ndarray:
        """
        Convert integer state to normalized (x,y) coordinates.

        Args:
            state (int): The current state number

        Returns:
            np.ndarray: Normalized [x, y] coordinates between 0 and 1
        """
        # Convert state number to x,y coordinates
        y = state // self.width
        x = state % self.width

        # Normalize coordinates
        norm_x = x / (self.width - 1)
        norm_y = y / (self.height - 1)

        return np.array([norm_x, norm_y], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment and return normalized state."""
        state, info = self.env.reset(seed=seed, options=options)
        self.steps=0
        return self._normalize_state(state), info

    def step(self, action: int) -> tuple[ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take action and return normalized next state."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.steps+=1
        if terminated:
            reward = 1 / self.steps
        else:
            reward = 0.

        if self.steps > 100: truncated = True
        return self._normalize_state(next_state), reward, terminated, truncated, info

    def gt(self, max_dist_accuracy=None):

        """
        Calculate ground truth distances for CliffWalking environment with normalized (x,y) coordinates.
        We always start from the bottom left corner (3, 0), the goal is the bottom right corner (3, 11).
        States from (3, 1) to (3, 10) are cliff states.

        Returns:
            Tuple of (s1_states, s2_states, d_gt) where states are normalized (x,y) coordinates
        """

        assert self.p == 1.0, "Ground truth calculation only supported for p=1.0"

        import itertools
        import torch

        height, width = self.height, self.width  # Standard CliffWalking dimensions
        start_pos = (height - 1, 0)  # (3,0)
        goal_pos = (height - 1, width - 1)  # (3,11)
        # Define cliff positions (bottom row except start and goal)
        cliff_states = [(height - 1, j) for j in range(1, width - 1)]

        def normalize_pos(pos):
            """Convert (i,j) position to normalized (x,y) coordinates."""
            i, j = pos
            return np.array([j / (width - 1), i / (height - 1)], dtype=np.float32)

        def get_manhattan_dist(pos1, pos2):
            """Calculate Manhattan distance between two positions."""
            i1, j1 = pos1
            i2, j2 = pos2
            return abs(i1 - i2) + abs(j1 - j2)

        def get_dist_through_cliff(pos1, pos2):
            """Calculate distance when going through closest cliff and back to start."""
            i1, j1 = pos1
            i2, j2 = pos2

            # Find closest cliff state to pos1
            min_dist_to_cliff = float('inf')
            for cliff_pos in cliff_states:
                dist = get_manhattan_dist((i1, j1), cliff_pos)
                min_dist_to_cliff = min(min_dist_to_cliff, dist)

            # After hitting cliff, we go from start to pos2
            dist_from_start = get_manhattan_dist(start_pos, (i2, j2))

            return min_dist_to_cliff + dist_from_start

        # Generate valid positions (excluding goal and cliffs)
        valid_positions = []
        for i in range(height):
            for j in range(width):
                if (i, j) != goal_pos and (i, j) not in cliff_states:
                    valid_positions.append((i, j))

        # Generate all pairs of valid positions
        all_pairs = list(itertools.product(valid_positions, valid_positions))

        # Calculate distances and normalize positions
        d_gt = []
        s1_normalized = []
        s2_normalized = []

        for pos1, pos2 in all_pairs:
            # Regular manhattan distance
            manhattan_dist = get_manhattan_dist(pos1, pos2)

            # Distance going through cliff
            cliff_dist = get_dist_through_cliff(pos1, pos2)

            # Take minimum of the two possible paths
            d_gt.append(min(manhattan_dist, cliff_dist))

            # Convert positions to normalized coordinates
            s1_normalized.append(normalize_pos(pos1))
            s2_normalized.append(normalize_pos(pos2))

        # Convert to PyTorch tensors
        s1_states = torch.FloatTensor(s1_normalized)
        s2_states = torch.FloatTensor(s2_normalized)
        d_gt = torch.FloatTensor(d_gt)

        if max_dist_accuracy is not None:
            d_gt = d_gt.clamp(max=max_dist_accuracy)

        return s1_states, s2_states, d_gt

    def starting_state_gt(self, max_dist_accuracy=None):
        """
        Calculate ground truth distances from the starting state to all valid states.
        Starting state is bottom left corner (3, 0).

        Args:
            max_dist_accuracy (float, optional): Maximum distance to clamp to

        Returns:
            Tuple of (s1_states, s2_states, d_gt) where:
                s1_states: tensor of normalized (x,y) coordinates for start state, repeated
                s2_states: tensor of normalized (x,y) coordinates for destination states
                d_gt: tensor of ground truth distances from start state to each destination
        """
        assert self.p == 1.0, "Ground truth calculation only supported for p=1.0"

        import torch

        height, width = self.height, self.width
        start_pos = (height - 1, 0)  # (3,0)
        goal_pos = (height - 1, width - 1)  # (3,11)
        cliff_states = [(height - 1, j) for j in range(1, width - 1)]

        def normalize_pos(pos):
            i, j = pos
            return np.array([j / (width - 1), i / (height - 1)], dtype=np.float32)

        def get_manhattan_dist(pos1, pos2):
            i1, j1 = pos1
            i2, j2 = pos2
            return abs(i1 - i2) + abs(j1 - j2)

        def get_dist_through_cliff(pos2):
            i2, j2 = pos2
            return 1 + get_manhattan_dist(start_pos, (i2, j2))

        valid_positions = []
        for i in range(height):
            for j in range(width):
                if (i, j) not in cliff_states:
                    valid_positions.append((i, j))

        d_gt = []
        s2_normalized = []

        for pos2 in valid_positions:
            manhattan_dist = get_manhattan_dist(start_pos, pos2)
            cliff_dist = get_dist_through_cliff(pos2)
            d_gt.append(min(manhattan_dist, cliff_dist))
            s2_normalized.append(normalize_pos(pos2))

        # Convert to tensors
        s2_states = torch.FloatTensor(s2_normalized)
        d_gt = torch.FloatTensor(d_gt)

        # Create s1_states tensor with repeated start state
        start_normalized = normalize_pos(start_pos)
        s1_states = torch.FloatTensor(start_normalized).unsqueeze(0).repeat(len(valid_positions), 1)

        if max_dist_accuracy is not None:
            d_gt = d_gt.clamp(max=max_dist_accuracy)

        return s1_states, s2_states, d_gt
