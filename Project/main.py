import pygame
import random
import numpy as np
import time
import gymnasium as gym  # Use Gymnasium API
from gymnasium import spaces

# --- Constants ---
# Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4 # Optional: Allow agent to stay still (useful near charger)
NUM_ACTIONS = 5 # 4 if ACTION_STAY is removed

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

class RoombaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30} # Adjusted FPS for smoother rendering

    def __init__(self, width=800, height=600, render_mode=None, max_steps=2000, num_dirt=100):
        super().__init__()

        self.width = width
        self.height = height
        self.max_dist = np.sqrt(width**2 + height**2) # Max possible distance for normalization

        # Roomba properties
        self.roomba_radius = 20
        self.roomba_speed = 5.0 # Make float for consistency
        self.initial_roomba_pos = np.array([width // 2, height // 2], dtype=np.float32)
        self.roomba_pos = self.initial_roomba_pos.copy()

        # Battery properties
        self.initial_battery = 100.0
        self.battery_level = self.initial_battery
        self.battery_decrease_per_step = 0.05 # Decrease battery slightly each step moved
        self.charge_increase_per_step = 0.2  # Charge rate per step while on station
        self.low_battery_threshold = 20 # Threshold to incentivize charging

        # Charging station
        self.charging_station_pos = np.array([50, 50], dtype=np.float32)
        self.charging_station_size = 40

        # Walls
        self.walls = []
        self._generate_walls() # Use underscore for internal setup methods

        # Dirt particles
        self.initial_dirt_count = num_dirt
        self.dirt_particles = []
        self.dirt_radius = 3
        self._generate_dirt(self.initial_dirt_count)

        # Reward system
        self.score = 0 # Cumulative score within an episode
        self.reward_per_dirt = 10.0
        self.penalty_per_step = -0.01 # Small penalty for time passing
        self.penalty_battery_empty = -50.0 # Large penalty for running out of battery
        self.reward_fully_charged = 5.0 # Small reward for reaching full charge
        self.reward_for_charging_when_low = 0.1 # Incentive to charge when low

        # Episode termination
        self.max_episode_steps = max_steps
        self.current_step = 0

        # --- Define Action and Observation Space ---
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Observations:
        # 1. Roomba X (normalized)
        # 2. Roomba Y (normalized)
        # 3. Battery Level (normalized)
        # 4. Distance to Charging Station (normalized)
        # 5. Angle to Charging Station (normalized to -1 to 1)
        # 6. Distance to Nearest Dirt (normalized)
        # 7. Angle to Nearest Dirt (normalized to -1 to 1)
        # 8. Number of dirt particles remaining (normalized)
        obs_dim = 8
        low = np.array([0.0] * obs_dim, dtype=np.float32)
        high = np.array([1.0] * obs_dim, dtype=np.float32)
        # Adjust bounds for angles
        low[4], high[4] = -1.0, 1.0 # Angle to station
        low[6], high[6] = -1.0, 1.0 # Angle to dirt

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Pygame Setup (only if rendering) ---
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Roomba RL Environment")
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)


    def _generate_walls(self):
        """Generate walls in the room"""
        wall_width = 20
        # Outer walls
        self.walls = [
            pygame.Rect(0, 0, self.width, wall_width),  # Top
            pygame.Rect(0, self.height - wall_width, self.width, wall_width),  # Bottom
            pygame.Rect(0, 0, wall_width, self.height),  # Left
            pygame.Rect(self.width - wall_width, 0, wall_width, self.height),  # Right
        ]
        # Add some inner walls
        self.walls.extend(
            [
                pygame.Rect(200, 150, wall_width, 300),  # Vertical wall
                pygame.Rect(400, 200, 200, wall_width),  # Horizontal wall
                pygame.Rect(600, 400, wall_width, 200),  # Another vertical wall
            ]
        )

    def _generate_dirt(self, num_particles):
        """Generate random dirt particles avoiding walls"""
        self.dirt_particles = []
        attempts = 0
        max_attempts = num_particles * 10 # Prevent infinite loop if space is too crowded

        while len(self.dirt_particles) < num_particles and attempts < max_attempts:
            x = random.uniform(self.dirt_radius, self.width - self.dirt_radius)
            y = random.uniform(self.dirt_radius, self.height - self.dirt_radius)
            # Represent dirt as numpy arrays for easier calculations
            new_dirt_pos = np.array([x, y], dtype=np.float32)

            # Check collision with walls
            dirt_rect = pygame.Rect(
                x - self.dirt_radius, y - self.dirt_radius,
                self.dirt_radius * 2, self.dirt_radius * 2
            )
            if not any(wall.colliderect(dirt_rect) for wall in self.walls):
                 # Check collision with charging station
                 station_rect = pygame.Rect(
                     self.charging_station_pos[0] - self.charging_station_size / 2,
                     self.charging_station_pos[1] - self.charging_station_size / 2,
                     self.charging_station_size, self.charging_station_size
                 )
                 if not station_rect.colliderect(dirt_rect):
                     self.dirt_particles.append(new_dirt_pos)
            attempts += 1
        if len(self.dirt_particles) < num_particles:
             print(f"Warning: Could only generate {len(self.dirt_particles)} dirt particles out of {num_particles} requested.")


    def _get_obs(self):
        """Calculate the current observation state."""
        # 1 & 2: Normalized Roomba Position
        norm_pos_x = self.roomba_pos[0] / self.width
        norm_pos_y = self.roomba_pos[1] / self.height

        # 3: Normalized Battery Level
        norm_battery = self.battery_level / self.initial_battery

        # 4 & 5: Distance and Angle to Charging Station
        vec_to_station = self.charging_station_pos - self.roomba_pos
        dist_to_station = np.linalg.norm(vec_to_station)
        norm_dist_to_station = min(dist_to_station / self.max_dist, 1.0) # Cap at 1
        angle_to_station = np.arctan2(vec_to_station[1], vec_to_station[0]) / np.pi # Normalize to [-1, 1]

        # 6 & 7: Distance and Angle to Nearest Dirt
        if not self.dirt_particles:
            norm_dist_to_dirt = 1.0 # Indicate no dirt nearby / max distance
            angle_to_dirt = 0.0
        else:
            dirt_positions = np.array(self.dirt_particles)
            vectors_to_dirt = dirt_positions - self.roomba_pos
            distances_to_dirt = np.linalg.norm(vectors_to_dirt, axis=1)
            nearest_dirt_idx = np.argmin(distances_to_dirt)
            nearest_dirt_vec = vectors_to_dirt[nearest_dirt_idx]
            dist_to_nearest_dirt = distances_to_dirt[nearest_dirt_idx]

            norm_dist_to_dirt = min(dist_to_nearest_dirt / self.max_dist, 1.0) # Cap at 1
            angle_to_dirt = np.arctan2(nearest_dirt_vec[1], nearest_dirt_vec[0]) / np.pi # Normalize to [-1, 1]

        # 8: Normalized remaining dirt count
        norm_dirt_count = len(self.dirt_particles) / self.initial_dirt_count

        obs = np.array([
            norm_pos_x, norm_pos_y, norm_battery,
            norm_dist_to_station, angle_to_station,
            norm_dist_to_dirt, angle_to_dirt,
            norm_dirt_count
        ], dtype=np.float32)

        # Ensure observation is within bounds (important due to potential float inaccuracies)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        return obs

    def _get_info(self):
        """Return auxiliary information (optional)."""
        return {
            "distance_to_station": np.linalg.norm(self.charging_station_pos - self.roomba_pos),
            "dirt_remaining": len(self.dirt_particles),
            "score": self.score,
            "battery": self.battery_level,
            "steps": self.current_step,
        }

    def _is_at_charging_station(self):
        """Check if Roomba center is within the charging station radius."""
        # Use a slightly larger radius check for robustness
        return np.linalg.norm(self.roomba_pos - self.charging_station_pos) < (self.charging_station_size / 2)

    def _check_collision(self, pos):
        """Check if position collides with walls."""
        roomba_rect = pygame.Rect(
            pos[0] - self.roomba_radius, pos[1] - self.roomba_radius,
            self.roomba_radius * 2, self.roomba_radius * 2
        )
        return any(wall.colliderect(roomba_rect) for wall in self.walls)

    def _clean_dirt(self):
        """Clean dirt particles under the Roomba. Returns number cleaned."""
        cleaned_count = 0
        if self.battery_level > 0: # Can only clean if not dead
            # More efficient check: iterate through dirt near the roomba
            roomba_sq_radius = (self.roomba_radius + self.dirt_radius) ** 2
            remaining_dirt = []
            for dirt_pos in self.dirt_particles:
                dist_sq = np.sum((self.roomba_pos - dirt_pos)**2)
                if dist_sq <= roomba_sq_radius:
                    cleaned_count += 1
                    self.score += self.reward_per_dirt
                else:
                    remaining_dirt.append(dirt_pos)
            self.dirt_particles = remaining_dirt
        return cleaned_count

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed) # Important for reproducibility

        # Reset state variables
        self.roomba_pos = self.initial_roomba_pos.copy()
        self.battery_level = self.initial_battery
        self.current_step = 0
        self.score = 0

        # Regenerate dirt
        self._generate_dirt(self.initial_dirt_count)

        observation = self._get_obs()
        info = self._get_info()

        # Reset rendering if needed (clears screen for new episode)
        # if self.render_mode == "human":
        #     self._render_frame() # Render initial frame

        return observation, info

    def step(self, action):
        """Performs one step in the environment."""
        self.current_step += 1
        reward = 0.0
        moved = False

        # --- 1. Apply Action ---
        if self.battery_level > 0: # Can only move if battery isn't dead
            delta_pos = np.array([0.0, 0.0], dtype=np.float32)
            if action == ACTION_UP:
                delta_pos[1] = -self.roomba_speed
                moved = True
            elif action == ACTION_DOWN:
                delta_pos[1] = self.roomba_speed
                moved = True
            elif action == ACTION_LEFT:
                delta_pos[0] = -self.roomba_speed
                moved = True
            elif action == ACTION_RIGHT:
                delta_pos[0] = self.roomba_speed
                moved = True
            elif action == ACTION_STAY:
                moved = False # Explicitly not moving

            if moved:
                new_pos = self.roomba_pos + delta_pos
                if not self._check_collision(new_pos):
                    self.roomba_pos = new_pos
                #else: # Optional: small penalty for hitting a wall?
                #    reward -= 0.5

        # --- 2. Update Environment State (Battery, Charging) ---
        at_station = self._is_at_charging_station()

        if at_station:
            # Charge battery if at charging station
            old_battery = self.battery_level
            self.battery_level = min(self.initial_battery, self.battery_level + self.charge_increase_per_step)
            if old_battery < self.initial_battery and self.battery_level == self.initial_battery:
                 reward += self.reward_fully_charged # Reward reaching full charge
            if old_battery < self.low_battery_threshold:
                 reward += self.reward_for_charging_when_low # Encourage charging when low
        elif moved:
            # Decrease battery if moved and not charging
            self.battery_level = max(0, self.battery_level - self.battery_decrease_per_step)

        # --- 3. Clean Dirt ---
        dirt_cleaned_this_step = self._clean_dirt()
        reward += dirt_cleaned_this_step * self.reward_per_dirt

        # --- 4. Calculate Reward ---
        reward += self.penalty_per_step # Apply time penalty

        # --- 5. Check Termination Conditions ---
        terminated = False
        truncated = False

        if self.battery_level <= 0:
            reward += self.penalty_battery_empty
            terminated = True
            print(f"Episode terminated: Battery empty after {self.current_step} steps.")


        if not self.dirt_particles:
            # Optional: Give bonus for finishing? Maybe handled by cumulative reward.
            # reward += 50.0 # Example bonus
            terminated = True
            print(f"Episode terminated: All dirt cleaned after {self.current_step} steps. Score: {self.score}")


        if self.current_step >= self.max_episode_steps:
            truncated = True
            print(f"Episode truncated: Max steps ({self.max_episode_steps}) reached.")


        # --- 6. Get Observation and Info ---
        observation = self._get_obs()
        info = self._get_info()

        # --- 7. Render (Optional) ---
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info


    def render(self):
       """Required by Gym API, calls internal rendering function."""
       if self.render_mode == "human":
            return self._render_frame()
       # Could add other modes like "rgb_array" later
       return None


    def _render_frame(self):
        """Renders the current state using Pygame."""
        if self.screen is None:
             # Initialize Pygame if not already done (e.g., if render() called before reset())
             pygame.init()
             pygame.display.set_caption("Roomba RL Environment")
             self.screen = pygame.display.set_mode((self.width, self.height))
             self.clock = pygame.time.Clock()
             self.font = pygame.font.Font(None, 36)

        self.screen.fill(WHITE)

        # Draw charging station
        station_rect = pygame.Rect(
            self.charging_station_pos[0] - self.charging_station_size / 2,
            self.charging_station_pos[1] - self.charging_station_size / 2,
            self.charging_station_size,
            self.charging_station_size,
        )
        pygame.draw.rect(self.screen, YELLOW, station_rect)
        pygame.draw.rect(self.screen, BLACK, station_rect, 2) # Outline

        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, GRAY, wall)

        # Draw dirt particles
        for dirt_pos in self.dirt_particles:
            pygame.draw.circle(
                self.screen, BROWN, (int(dirt_pos[0]), int(dirt_pos[1])), self.dirt_radius
            )

        # Draw Roomba with color based on battery level
        if self.battery_level > 50:
            color = BLACK
        elif self.battery_level > self.low_battery_threshold:
            color = (180, 180, 0) # Darker Yellow
        else:
            color = RED

        pygame.draw.circle(
            self.screen,
            color,
            (int(self.roomba_pos[0]), int(self.roomba_pos[1])),
            self.roomba_radius,
        )
        # Draw outline for visibility
        pygame.draw.circle(
            self.screen,
            GRAY,
            (int(self.roomba_pos[0]), int(self.roomba_pos[1])),
            self.roomba_radius,
            1
        )


        # --- Display Information ---
        info_y = 10
        info_step = 30

        # Score
        score_text = self.font.render(f"Score: {self.score:.0f}", True, BLACK)
        self.screen.blit(score_text, (10, info_y))
        info_y += info_step

        # Battery Level
        battery_text = self.font.render(f"Battery: {self.battery_level:.1f}%", True, BLACK)
        batt_col = GREEN if self.battery_level > 50 else YELLOW if self.battery_level > self.low_battery_threshold else RED
        pygame.draw.rect(self.screen, GRAY, (10, info_y + 25, 150, 15)) # Background bar
        pygame.draw.rect(self.screen, batt_col, (10, info_y + 25, 150 * (self.battery_level/self.initial_battery), 15)) # Fill bar
        self.screen.blit(battery_text, (10, info_y))
        info_y += info_step + 15 # Extra space for bar

        # Dirt Remaining
        dirt_text = self.font.render(f"Dirt Remaining: {len(self.dirt_particles)}", True, BLACK)
        self.screen.blit(dirt_text, (10, info_y))
        info_y += info_step

         # Steps
        step_text = self.font.render(f"Step: {self.current_step}/{self.max_episode_steps}", True, BLACK)
        self.screen.blit(step_text, (10, info_y))
        info_y += info_step

        # Charging Indicator
        if self._is_at_charging_station():
            charge_indicator = "CHARGING"
            charge_color = GREEN
            if self.battery_level == self.initial_battery:
                 charge_indicator = "CHARGED"
                 charge_color = BLUE # Use a different color for fully charged
            charging_text = self.font.render(charge_indicator, True, charge_color)
            self.screen.blit(charging_text, (10, info_y))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"]) # Control framerate

    def close(self):
        """Clean up resources (Pygame window)."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


# --- Example Usage ---
if __name__ == "__main__":
    # To run with rendering:
    env = RoombaEnv(render_mode="human", max_steps=1500, num_dirt=50)

    # To run without rendering (faster for training):
    # env = RoombaEnv(render_mode=None, max_steps=1500, num_dirt=50)

    # --- Basic Interaction Loop (Random Agent) ---
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)
    print("Observation Space Sample:", env.observation_space.sample())


    episodes = 3
    for episode in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        while not terminated and not truncated:
            # --- Replace with your RL agent's action selection ---
            action = env.action_space.sample() # Random action
            # ----------------------------------------------------

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Optional: Render frame even if render_mode wasn't set initially
            # env.render() # Uncomment to force rendering frames

            if terminated or truncated:
                print(f"Episode {episode + 1}: Finished!")
                print(f"  Total Reward: {total_reward:.2f}")
                print(f"  Steps: {step_count}")
                print(f"  Final Info: {info}")
                print("-" * 20)

            # Add a small delay for visualization if rendering
            # if env.render_mode == "human":
            #    time.sleep(0.01)


    env.close()
    print("Environment closed.")