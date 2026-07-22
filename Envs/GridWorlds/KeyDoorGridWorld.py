import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class KeyDoorGridWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 800}

    def __init__(self, render_mode=None, max_steps=100, dense_reward=False, p=1.0):
        size = 13  # The size of the square grid
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.max_steps = max_steps
        self.dense_reward = dense_reward
        self.p = p  # Probability of the action succeeding

        # Observation space: (x, y, has_key)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]),
                                            high=np.array([1, 1, 1]),
                                            dtype=np.int32)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, -1]),  # Note: in grid coordinates, up is -1
            2: np.array([-1, 0]),
            3: np.array([0, 1]),  # Note: in grid coordinates, down is +1
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # Initialize the grid
        self.grid = np.zeros((size, size), dtype=int)

        # Set fixed positions
        self._agent_location = np.array([1, 1])
        self._key_location = np.array([11, 1])
        self._door_location = np.array([6, 9])

        self.has_key = False
        self.door_open = False
        self.steps = 0
        self.seed = None

    def _get_obs(self):
        return np.array([self._agent_location[0]/self.size, self._agent_location[1]/self.size, int(self.has_key)])

    def _get_info(self):
        return {
            "distance_to_key": np.linalg.norm(self._agent_location - self._key_location,
                                              ord=1) if not self.has_key else 0,
            "door_open": self.door_open,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        if self.seed is None:
            self.seed = seed
            super().reset(seed=seed)
            self.action_space.seed(seed)

        self._agent_location = np.array([1, 1])
        self._key_location = np.array([11, 1])
        self._door_location = np.array([6, 9])
        self.has_key = False
        self.door_open = False
        self.steps = 0

        # Reset door to closed state
        self.grid[self._door_location[0], self._door_location[1]] = 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.steps += 1

        reward = 0
        terminated = False
        truncated = self.steps >= self.max_steps

        if self.np_random.random() < self.p:
            direction = self._action_to_direction[action]
        else:
            direction = self._action_to_direction[self.np_random.integers(0, 4)]

        new_position = self._agent_location + direction

        if (0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size and
                (self.grid[new_position[0], new_position[1]] == 0)):

            # Move the agent
            self._agent_location = new_position

        # Check for key pickup
        if np.array_equal(self._agent_location, self._key_location) and not self.has_key:
            self.has_key = True
            self._key_location = np.array([-1, -1])  # Move key off-grid
            if self.dense_reward: reward += 1

        if np.array_equal(new_position, self._door_location):

            # Check for door opening
            if self.has_key and not self.door_open:
                self.door_open = True
                self.grid[self._door_location[0], self._door_location[1]] = 0  # Open the door in the grid
                reward += 1

            if self.door_open:
                self._agent_location = new_position
                terminated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Initialize font
        font = pygame.font.Font(None, 30)

        # Draw the grid
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 1:  # wall
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),  # Black color for walls
                        pygame.Rect(
                            pix_square_size * np.array([i, j]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Draw the key (K)
        if not self.has_key:
            pygame.draw.rect(
                canvas,
                (0, 255, 255),  # Cyan color for the key
                pygame.Rect(
                    pix_square_size * self._key_location,
                    (pix_square_size, pix_square_size),
                ),
            )
            text = font.render('K', True, (0, 0, 0))
            text_rect = text.get_rect(center=(pix_square_size * (self._key_location[0] + 0.5),
                                              pix_square_size * (self._key_location[1] + 0.5)))
            canvas.blit(text, text_rect)

        # Draw the door (D)
        if not self.door_open:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),  # Green color for the door
                pygame.Rect(
                    pix_square_size * self._door_location,
                    (pix_square_size, pix_square_size),
                ),
            )
            text = font.render('D', True, (0, 0, 0))
            text_rect = text.get_rect(center=(pix_square_size * (self._door_location[0] + 0.5),
                                              pix_square_size * (self._door_location[1] + 0.5)))
            canvas.blit(text, text_rect)

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Add gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),  # Light gray for grid lines
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (200, 200, 200),  # Light gray for grid lines
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def gt(self, max_dist_accuracy=None):
        assert self.p == 1, "GT only works for deterministic environments (p=1)"

        import torch

        size = int(self.size)
        obj_pos = torch.FloatTensor(self._key_location / size)

        # Calculate same state pairs
        same_state_pairs = []
        for i in range(size):
            for j in range(size):
                if i != self._key_location[0] and j != self._key_location[1]:
                    same_state_pairs.append([(i / size, j / size, 0), (i / size, j / size, 1)])

        same_state_pairs = torch.FloatTensor(same_state_pairs)
        s1_same = same_state_pairs[:, 0]
        s2_same = same_state_pairs[:, 1]

        # Calculate ground truth for same state pairs
        to_obj_same = torch.norm(s1_same[:, :2] - obj_pos, 1, dim=1) * size
        from_obj_same = torch.norm(s2_same[:, :2] - obj_pos, 1, dim=1) * size
        d_gt_same = to_obj_same + from_obj_same

        # Same state pairs reversed
        s1_same_reversed = same_state_pairs[:, 1]
        s2_same_reversed = same_state_pairs[:, 0]
        if max_dist_accuracy is not None:
            d_gt_same_reverse = torch.ones(len(s1_same_reversed)) * max_dist_accuracy
        else:
            d_gt_same_reverse = torch.ones(len(s1_same_reversed)) * float("inf")

        # Without object pairs
        wo_obj_pairs = []
        for i in range(size):
            for j in range(size):
                wo_obj_pairs.append([(i / size, j / size, 0), (j / size, j / size, 0)])

        wo_obj_pairs = torch.FloatTensor(wo_obj_pairs)
        s1_wo_obj = wo_obj_pairs[:, 0]
        s2_wo_obj = wo_obj_pairs[:, 1]
        d_gt_wo_obj = torch.norm(s1_wo_obj[:, :2] - s2_wo_obj[:, :2], 1, dim=1) * size

        # With object pairs
        w_obj_pairs = []
        for i in range(size):
            for j in range(size):
                w_obj_pairs.append([(i / size, j / size, 1), (j / size, j / size, 1)])

        w_obj_pairs = torch.FloatTensor(w_obj_pairs)
        s1_w_obj = w_obj_pairs[:, 0]
        s2_w_obj = w_obj_pairs[:, 1]
        d_gt_w_obj = torch.norm(s1_w_obj[:, :2] - s2_w_obj[:, :2], 1, dim=1) * size

        if max_dist_accuracy is not None:
            d_gt_same = torch.min(d_gt_same, torch.ones(len(d_gt_same)) * max_dist_accuracy)
            d_gt_wo_obj = torch.min(d_gt_wo_obj, torch.ones(len(d_gt_wo_obj)) * max_dist_accuracy)
            d_gt_w_obj = torch.min(d_gt_w_obj, torch.ones(len(d_gt_w_obj)) * max_dist_accuracy)

        return (torch.concat([s1_same, s1_same_reversed, s1_wo_obj, s1_w_obj]),
                torch.concat([s2_same, s2_same_reversed, s2_wo_obj, s2_w_obj]),
                torch.concat([d_gt_same, d_gt_same_reverse, d_gt_wo_obj, d_gt_w_obj]))

    def human_play(self):
        """
        Allows human to play the game using arrow keys.
        """
        if self.render_mode != "human":
            raise ValueError("To play the game manually, please set render_mode='human'")

        observation, info = self.reset(seed=0)
        self._render_frame()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 3
                    elif event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 0
                    else:
                        continue

                    observation, reward, terminated, truncated, info = self.step(action)
                    print(observation, reward)
                    self._render_frame()

                    if terminated:
                        print("Episode terminated")
                        running = False

                    if truncated:
                        print("Episode truncated")
                        running = False

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        self.close()


# Example usage
if __name__ == "__main__":
    env = KeyDoorGridWorld(render_mode="human", dense_reward=True, p=1)
    env.human_play()