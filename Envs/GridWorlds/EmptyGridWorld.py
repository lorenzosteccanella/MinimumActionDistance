import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class EmptyGridWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 800}

    def __init__(self, render_mode=None, max_steps=100, dense_reward=False, p=0.5):
        size = 13  # The size of the square grid
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.max_steps = max_steps
        self.dense_reward = dense_reward
        self.p = p  # Probability of the action succeeding

        # Observation space: (x, y)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]),
                                            high=np.array([1, 1, 1, 1]),
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

        self.steps = 0
        self.seed = None

    def _get_obs(self):

        # let's add some noise to the observation
        obs = np.array([self._agent_location[0]/self.size, self._agent_location[1]/self.size, np.random.random(), np.random.random()])

        return obs

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        if self.seed is None:
            self.seed = seed
            super().reset(seed=seed)
            self.action_space.seed(seed)

        self._agent_location = (self.np_random.random(2) * self.size) + self.np_random.random(2)
        self.steps = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.steps += 1

        terminated = False
        truncated = self.steps >= self.max_steps

        if self.np_random.random() < self.p:
            direction = self._action_to_direction[action]
        else:
            direction = self._action_to_direction[self.np_random.integers(0, 4)]

        new_position = (self._agent_location + direction)

        if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size:

            # Move the agent
            self._agent_location = new_position

        reward = 0.

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

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
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
        import torch
        import itertools

        all_states = np.array([(i, j) for i in range(self.size) for j in range(self.size)])
        random_offset = np.random.rand(len(all_states), 2)
        all_states = all_states + random_offset
        all_states = all_states / self.size

        # Add random noise dimensions to match observation space
        noise = np.random.rand(len(all_states), 2)  # Two random dimensions
        all_states = np.concatenate([all_states, noise], axis=1)  # Now it's (x, y, noise1, noise2)

        all_pairs = np.array(list(itertools.product(all_states, all_states)))
        s1_gt = torch.FloatTensor(all_pairs[:, 0])
        s2_gt = torch.FloatTensor(all_pairs[:, 1])

        # Calculate L1 distance using only the first two dimensions (x,y)
        d_gt = torch.norm(s1_gt[:, :2] - s2_gt[:, :2], 1, dim=1) * self.size

        if max_dist_accuracy is not None:
            d_gt = d_gt.clamp(max=max_dist_accuracy)

        return s1_gt, s2_gt, d_gt

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
                    print(observation, self._agent_location, reward)
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
    env = EmptyGridWorld(render_mode="human", dense_reward=False)
    env.human_play()