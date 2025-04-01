import numpy as np
import random
import pygame
import sys
import time
import os
import heapq
from collections import defaultdict
import threading

class RobotVacuumEnvironment:
    """Environment for the robot vacuum cleaner simulation with enhanced mapping capabilities"""
    
    # Action space: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
    ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    ACTION_NAMES = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    
    # Cell types
    UNKNOWN = 0
    EMPTY = 1
    OBSTACLE = 2
    ROBOT = 3
    ROOM_BOUNDARY = 4  # New type for room boundaries
    
    def __init__(self, width=15, height=12, obstacle_prob=0.2):
        self.width = width
        self.height = height
        self.obstacle_prob = obstacle_prob
        
        # Create the actual house layout (hidden from robot initially)
        self.house_layout = np.ones((height, width))  # Start with all empty cells
        
        # Add random obstacles
        for y in range(height):
            for x in range(width):
                if random.random() < obstacle_prob:
                    self.house_layout[y, x] = self.OBSTACLE
        
        # Ensure starting position is empty
        self.house_layout[height // 2, width // 2] = self.EMPTY
        
        # Initialize robot position
        self.robot_pos = (width // 2, height // 2)
        
        # Robot's knowledge map (what it has discovered)
        self.knowledge_map = np.zeros((height, width))
        
        # Traffic density map (how often each cell is visited)
        self.traffic_map = np.zeros((height, width))
        
        # Time efficiency map (how long it takes to navigate to each cell)
        self.time_efficiency_map = np.zeros((height, width))
        
        # Store detected rooms
        self.room_map = np.zeros((height, width), dtype=int)  # Will store room IDs
        self.next_room_id = 1
        
        # Update the starting position in the knowledge map
        self.knowledge_map[self.robot_pos[1], self.robot_pos[0]] = self.EMPTY
        
        # Update traffic at starting position
        self.traffic_map[self.robot_pos[1], self.robot_pos[0]] += 1
        
        # Track the number of moves
        self.moves = 0
        
        # Track the coverage (percentage of non-obstacle cells discovered)
        self.coverage = self.calculate_coverage()
        
        # Direction the robot is facing (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        self.facing = 1  # Start facing right
        
        # Animation tracking
        self.last_action = None
        self.animation_progress = 0
        
        # Path history
        self.path_history = [(self.robot_pos[0], self.robot_pos[1])]
        
        # Optimal path data
        self.optimal_path = []
        self.target_position = None
        
        # Navigation timestamps for time efficiency calculation
        self.cell_timestamps = {}
        self.current_time = 0
        
        # For asynchronous path calculation
        self.path_calculation_thread = None
        self.calculation_lock = threading.Lock()
    
    def calculate_coverage(self):
        """Calculate the percentage of non-obstacle cells discovered"""
        total_explorable = np.sum(self.house_layout == self.EMPTY)
        explored = np.sum((self.knowledge_map == self.EMPTY) & (self.house_layout == self.EMPTY))
        return explored / total_explorable if total_explorable > 0 else 0
    
    def detect_rooms(self):
        """Detect separate rooms in the house layout using flood fill"""
        # Reset room map
        self.room_map = np.zeros((self.height, self.width), dtype=int)
        room_id = 1
        
        # Flood fill to identify connected components (rooms)
        for y in range(self.height):
            for x in range(self.width):
                if self.knowledge_map[y, x] == self.EMPTY and self.room_map[y, x] == 0:
                    self._flood_fill(x, y, room_id)
                    room_id += 1
        
        self.next_room_id = room_id
    
    def _flood_fill(self, x, y, room_id):
        """Helper function for room detection using flood fill"""
        queue = [(x, y)]
        while queue:
            cx, cy = queue.pop(0)
            if (0 <= cx < self.width and 0 <= cy < self.height and 
                self.knowledge_map[cy, cx] == self.EMPTY and 
                self.room_map[cy, cx] == 0):
                
                self.room_map[cy, cx] = room_id
                
                # Check adjacent cells
                for dx, dy in self.ACTIONS:
                    nx, ny = cx + dx, cy + dy
                    queue.append((nx, ny))
    
    def get_state(self):
        """Return the current state for the RL agent"""
        # Create a combined map for visualization
        state = self.knowledge_map.copy()
        state[self.robot_pos[1], self.robot_pos[0]] = self.ROBOT
        
        # For the RL agent, we return a more detailed state representation
        surroundings = []
        for action in self.ACTIONS:
            x, y = self.robot_pos[0] + action[0], self.robot_pos[1] + action[1]
            if 0 <= x < self.width and 0 <= y < self.height:
                surroundings.append(self.knowledge_map[y, x])
            else:
                surroundings.append(self.OBSTACLE)  # Treat edges as obstacles
        
        # Convert robot position to a normalized coordinate
        norm_x = self.robot_pos[0] / self.width
        norm_y = self.robot_pos[1] / self.height
        
        # Calculate the number of unknown cells in each direction (limited range)
        unknown_counts = []
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # Up, right, down, left
            count = 0
            for dist in range(1, 4):  # Look up to 3 cells ahead
                x, y = self.robot_pos[0] + dx * dist, self.robot_pos[1] + dy * dist
                if 0 <= x < self.width and 0 <= y < self.height and self.knowledge_map[y, x] == self.UNKNOWN:
                    count += 1
            unknown_counts.append(count)
        
        return {
            'visual_state': state,
            'surroundings': surroundings,
            'position': (norm_x, norm_y),
            'unknown_counts': unknown_counts,
            'coverage': self.coverage,
            'facing': self.facing,
            'room_map': self.room_map
        }
    
    def step(self, action):
        """Take an action and return new state, reward, and done flag"""
        # Update the direction the robot is facing
        self.facing = action
        self.last_action = action
        self.animation_progress = 0  # Reset animation
        
        new_x = self.robot_pos[0] + self.ACTIONS[action][0]
        new_y = self.robot_pos[1] + self.ACTIONS[action][1]
        
        # Update current time
        self.current_time += 1
        
        # Check if the move is valid
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            # Check if there's an obstacle in the actual house layout
            if self.house_layout[new_y, new_x] != self.OBSTACLE:
                # Valid move
                self.robot_pos = (new_x, new_y)
                
                # Update path history
                self.path_history.append((new_x, new_y))
                
                # Update traffic map with the new position
                self.traffic_map[new_y, new_x] += 1
                
                # Update time efficiency map
                if (new_x, new_y) not in self.cell_timestamps:
                    self.cell_timestamps[(new_x, new_y)] = self.current_time
                    self.time_efficiency_map[new_y, new_x] = self.current_time
                
                # Update knowledge map with the new cell
                newly_discovered = self.knowledge_map[new_y, new_x] == self.UNKNOWN
                self.knowledge_map[new_y, new_x] = self.EMPTY
                
                # Also update adjacent cells that can be seen
                for dx, dy in self.ACTIONS:
                    adj_x, adj_y = new_x + dx, new_y + dy
                    if 0 <= adj_x < self.width and 0 <= adj_y < self.height:
                        if self.knowledge_map[adj_y, adj_x] == self.UNKNOWN:
                            if self.house_layout[adj_y, adj_x] == self.OBSTACLE:
                                self.knowledge_map[adj_y, adj_x] = self.OBSTACLE
                            else:
                                self.knowledge_map[adj_y, adj_x] = self.EMPTY
                
                # Periodically update room detection
                if self.moves % 10 == 0:
                    self.detect_rooms()
                
                # Update optimal path if we have a target
                if newly_discovered and self.target_position is None:
                    self.update_exploration_target()
                
                # Increment move counter
                self.moves += 1
                
                # Calculate new coverage
                old_coverage = self.coverage
                self.coverage = self.calculate_coverage()
                coverage_increase = self.coverage - old_coverage
                
                # Calculate reward
                if newly_discovered:
                    # Reward for discovering new cells
                    reward = 1.0 + coverage_increase * 10
                else:
                    # Small penalty for revisiting known cells
                    reward = -0.1
                
                # Check if mapping is complete
                done = self.coverage >= 0.99  # Consider mapping complete at 99% coverage
                
                return self.get_state(), reward, done
            else:
                # Hit an obstacle
                self.knowledge_map[new_y, new_x] = self.OBSTACLE
                
                # If we had a path planned and hit an obstacle, recalculate
                if self.target_position is not None:
                    self.update_optimal_path()
                
                self.moves += 1
                return self.get_state(), -0.5, False  # Penalty for hitting an obstacle
        else:
            # Out of bounds
            self.moves += 1
            return self.get_state(), -0.5, False  # Penalty for trying to go out of bounds
    
    def update_exploration_target(self):
        """Find the nearest unexplored cell to target"""
        if self.path_calculation_thread is not None and self.path_calculation_thread.is_alive():
            return  # Don't start a new calculation if one is already running
        
        self.path_calculation_thread = threading.Thread(target=self._async_update_exploration_target)
        self.path_calculation_thread.daemon = True
        self.path_calculation_thread.start()
    
    def _async_update_exploration_target(self):
        """Asynchronously find the nearest unexplored cell to target"""
        with self.calculation_lock:
            # Find unexplored cells adjacent to known empty cells
            frontier_cells = []
            for y in range(self.height):
                for x in range(self.width):
                    if self.knowledge_map[y, x] == self.UNKNOWN:
                        # Check if adjacent to any known empty cell
                        for dx, dy in self.ACTIONS:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.width and 0 <= ny < self.height and 
                                self.knowledge_map[ny, nx] == self.EMPTY):
                                frontier_cells.append((x, y))
                                break
            
            if frontier_cells:
                # Sort by distance to current position
                frontier_cells.sort(key=lambda cell: 
                    ((cell[0] - self.robot_pos[0])**2 + (cell[1] - self.robot_pos[1])**2))
                
                # Take the closest frontier cell as target
                self.target_position = frontier_cells[0]
                self.update_optimal_path()
            else:
                self.target_position = None
                self.optimal_path = []
    
    def update_optimal_path(self):
        """Calculate optimal path to target using A* with traffic-based weighting"""
        if self.target_position is None:
            self.optimal_path = []
            return
        
        if self.path_calculation_thread is not None and self.path_calculation_thread.is_alive():
            return  # Don't start a new calculation if one is already running
            
        self.path_calculation_thread = threading.Thread(target=self._async_update_optimal_path)
        self.path_calculation_thread.daemon = True
        self.path_calculation_thread.start()
    
    def _async_update_optimal_path(self):
        """Asynchronously calculate optimal path using A* with traffic-based weighting"""
        with self.calculation_lock:
            if self.target_position is None:
                self.optimal_path = []
                return
            
            # A* pathfinding with custom weights
            start = self.robot_pos
            goal = self.target_position
            
            # Initialize data structures
            open_set = []
            heapq.heappush(open_set, (0, start))
            came_from = {}
            g_score = {start: 0}
            f_score = {start: self._heuristic(start, goal)}
            
            open_set_hash = {start}
            
            while open_set:
                # Pop cell with lowest f_score
                current = heapq.heappop(open_set)[1]
                open_set_hash.remove(current)
                
                # Goal reached
                if current == goal:
                    # Reconstruct path
                    path = []
                    while current in came_from:
                        path.append(current)
                        current = came_from[current]
                    path.append(start)
                    path.reverse()
                    
                    self.optimal_path = path
                    return
                
                # Explore neighbors
                for dx, dy in self.ACTIONS:
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # Check if valid move
                    if (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height and 
                        (self.knowledge_map[neighbor[1], neighbor[0]] == self.EMPTY or 
                         neighbor == goal)):
                        
                        # Calculate traffic-weighted cost
                        traffic_weight = max(1, self.traffic_map[neighbor[1], neighbor[0]])
                        move_cost = 1 + 0.1 * traffic_weight  # Higher traffic = higher cost
                        
                        tentative_g = g_score.get(current, float('inf')) + move_cost
                        
                        if tentative_g < g_score.get(neighbor, float('inf')):
                            # This path is better
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                            
                            if neighbor not in open_set_hash:
                                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                                open_set_hash.add(neighbor)
            
            # No path found
            self.optimal_path = []
            self.target_position = None
    
    def _heuristic(self, a, b):
        """Manhattan distance heuristic for A*"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_next_optimal_action(self):
        """Get the next action to follow the optimal path"""
        if not self.optimal_path or len(self.optimal_path) <= 1:
            return random.randint(0, 3)  # Random action if no path
        
        next_pos = self.optimal_path[1]  # Next position in path
        
        # Calculate action to take
        dx = next_pos[0] - self.robot_pos[0]
        dy = next_pos[1] - self.robot_pos[1]
        
        for i, (action_dx, action_dy) in enumerate(self.ACTIONS):
            if dx == action_dx and dy == action_dy:
                return i
        
        return random.randint(0, 3)  # Fallback


class QLearningAgent:
    """Q-learning agent for the robot vacuum"""
    
    def __init__(self, action_space=4, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        
        # Initialize Q-table as a dictionary for sparse state representation
        self.q_table = {}
    
    def get_state_key(self, state):
        """Convert state to a hashable key for the Q-table"""
        # Use relevant features from the state
        surroundings = tuple(state.get('surroundings', []))
        position = (round(state['position'][0] * 10) / 10, round(state['position'][1] * 10) / 10)
        unknown_counts = tuple(state.get('unknown_counts', []))
        
        return (surroundings, position, unknown_counts)
    
    def get_action(self, state):
        """Select an action using epsilon-greedy policy"""
        if random.random() < self.exploration_rate:
            # Exploration: choose a random action
            return random.randint(0, self.action_space - 1)
        else:
            # Exploitation: choose the best action from Q-table
            state_key = self.get_state_key(state)
            if state_key not in self.q_table:
                # If state not in Q-table, initialize it
                self.q_table[state_key] = [0] * self.action_space
            
            # Return action with highest Q-value (break ties randomly)
            best_value = max(self.q_table[state_key])
            best_actions = [a for a, q in enumerate(self.q_table[state_key]) if q == best_value]
            return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value for a state-action pair"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if not in table
        if state_key not in self.q_table:
            self.q_table[state_key] = [0] * self.action_space
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0] * self.action_space
        
        # Calculate Q-learning update
        if not done:
            next_max = max(self.q_table[next_state_key])
            new_q = self.q_table[state_key][action] + self.learning_rate * (
                reward + self.discount_factor * next_max - self.q_table[state_key][action]
            )
        else:
            new_q = self.q_table[state_key][action] + self.learning_rate * (
                reward - self.q_table[state_key][action]
            )
        
        self.q_table[state_key][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)


class PyGameVisualizer:
    """Visualizes the robot vacuum environment using Pygame"""
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    DARK_GRAY = (100, 100, 100)
    RED = (255, 50, 50)
    BLUE = (50, 50, 255)
    GREEN = (0, 255, 0)  # Bright green as requested
    YELLOW = (255, 255, 0)
    BUTTON_COLOR = (51, 102, 255)  # #3366FF as requested
    
    # Visualization modes
    MODE_NORMAL = 0
    MODE_TRAFFIC = 1
    MODE_OBSTACLE_DENSITY = 2
    MODE_TIME_EFFICIENCY = 3
    MODE_COUNT = 4
    
    def __init__(self, cell_size=40):
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        
        self.cell_size = cell_size
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 14)
        self.button_font = pygame.font.SysFont('Arial', 16, bold=True)
        
        # Robot images based on direction
        self.robot_images = self.load_robot_images()
        
        # Room colors for visualization
        self.room_colors = [
            (200, 200, 255),  # Light blue
            (200, 255, 200),  # Light green
            (255, 200, 200),  # Light red
            (255, 255, 200),  # Light yellow
            (255, 200, 255),  # Light purple
            (200, 255, 255),  # Light cyan
            (255, 220, 180),  # Light orange
            (220, 180, 255),  # Light violet
        ]
        
        # Placeholder for the screen
        self.screen = None
        self.width = 0
        self.height = 0
        
        # Path tracking
        self.path = []
        self.show_path = True
        
        # Button data
        self.button_rect = None
        self.visualization_mode = self.MODE_NORMAL
        
        # Legend data
        self.legend_rect = None
        
        # Animation variables
        self.animation_speed = 0.2  # Time (in seconds) for movement animation
        
        # Historical path data for episode comparison
        self.historical_paths = []
        self.historical_coverages = []
        
    def load_robot_images(self):
        """Create robot images for different directions"""
        images = []
        
        # Basic robot size
        robot_size = int(self.cell_size * 0.8)
        
        # Create a surface for each direction
        for i in range(4):  # UP, RIGHT, DOWN, LEFT
            surface = pygame.Surface((robot_size, robot_size), pygame.SRCALPHA)
            
            # Draw robot body (circle)
            pygame.draw.circle(surface, self.BLUE, (robot_size//2, robot_size//2), robot_size//2)
            
            # Draw direction indicator (triangle)
            if i == 0:  # UP
                points = [(robot_size//2, 0), (robot_size//4, robot_size//2), (3*robot_size//4, robot_size//2)]
            elif i == 1:  # RIGHT
                points = [(robot_size, robot_size//2), (robot_size//2, robot_size//4), (robot_size//2, 3*robot_size//4)]
            elif i == 2:  # DOWN
                points = [(robot_size//2, robot_size), (robot_size//4, robot_size//2), (3*robot_size//4, robot_size//2)]
            else:  # LEFT
                points = [(0, robot_size//2), (robot_size//2, robot_size//4), (robot_size//2, 3*robot_size//4)]
            
            pygame.draw.polygon(surface, self.YELLOW, points)
            images.append(surface)
        
        return images
    
    def setup(self, env):
        """Set up the visualization based on environment dimensions"""
        self.width = env.width
        self.height = env.height
        
        # Calculate window size with margins
        margin = 120
        window_width = self.width * self.cell_size + margin * 2
        window_height = self.height * self.cell_size + margin * 2
        
        # Create the window
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Robot Vacuum Mapping & Path Optimization")
        
        # Starting position for grid
        self.grid_x = margin
        self.grid_y = margin
        
        # Button position (bottom-right corner)
        self.button_rect = pygame.Rect(
            window_width - 130, 
            window_height - 60, 
            120, 40  # 120x40px as requested
        )
        
        # Legend position (bottom-left)
        self.legend_rect = pygame.Rect(
            margin, 
            window_height - 130,
            220, 120
        )
        
        # Reset path
        self.path = [env.robot_pos]
        
    def handle_events(self, env):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    self.show_path = not self.show_path
                elif event.key == pygame.K_n:
                    # Store data for the current episode before creating a new map
                    self.store_episode_data(env)
                    return "new_episode"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.button_rect.collidepoint(event.pos):
                    # Cycle to next visualization mode when button is clicked
                    self.visualization_mode = (self.visualization_mode + 1) % self.MODE_COUNT
        return None
    
    def store_episode_data(self, env):
        """Store path and coverage data from the current episode"""
        if len(env.path_history) > 0:
            self.historical_paths.append(env.path_history.copy())
            self.historical_coverages.append(env.coverage)
    
    def get_cell_color_by_mode(self, x, y, cell_type, env):
        """Get cell color based on the current visualization mode"""
        if cell_type == env.OBSTACLE:
            return self.BLACK
        
        if self.visualization_mode == self.MODE_NORMAL:
            # Normal mode: show rooms in different colors
            if cell_type == env.EMPTY:
                room_id = env.room_map[y, x]
                if room_id > 0:
                    return self.room_colors[(room_id - 1) % len(self.room_colors)]
                return self.WHITE
            return self.DARK_GRAY  # Unknown cells
            
        elif self.visualization_mode == self.MODE_TRAFFIC:
            # Traffic heat map mode
            if cell_type == env.EMPTY:
                # Scale traffic from blue (low) to red (high)
                traffic = env.traffic_map[y, x]
                max_traffic = np.max(env.traffic_map) if np.max(env.traffic_map) > 0 else 1
                intensity = min(1.0, traffic / max_traffic)
                r = int(255 * intensity)
                g = int(255 * (1 - intensity))
                b = int(255 * (1 - intensity))
                return (r, g, b)
            return self.DARK_GRAY
            
        elif self.visualization_mode == self.MODE_OBSTACLE_DENSITY:
            # Obstacle density map
            if cell_type == env.EMPTY:
                # Count obstacles in vicinity
                obstacle_count = 0
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < env.width and 0 <= ny < env.height and
                            env.knowledge_map[ny, nx] == env.OBSTACLE):
                            obstacle_count += 1
                
                # Scale from green (few obstacles nearby) to red (many obstacles)
                intensity = min(1.0, obstacle_count / 8) if obstacle_count > 0 else 0
                r = int(255 * intensity)
                g = int(255 * (1 - intensity))
                b = 50
                return (r, g, b)
            return self.DARK_GRAY
            
        elif self.visualization_mode == self.MODE_TIME_EFFICIENCY:
            # Time-based navigation efficiency
            if cell_type == env.EMPTY:
                # Color based on how quickly this cell was reached
                if (x, y) in env.cell_timestamps:
                    # Scale from green (early discovery) to red (late discovery)
                    timestamp = env.cell_timestamps[(x, y)]
                    max_time = env.current_time if env.current_time > 0 else 1
                    intensity = min(1.0, timestamp / max_time)
                    r = int(255 * intensity)
                    g = int(255 * (1 - intensity))
                    b = 100
                    return (r, g, b)
                return self.WHITE
            return self.DARK_GRAY
        
        # Default
        return self.WHITE if cell_type == env.EMPTY else self.DARK_GRAY
    
    def draw_cell(self, x, y, cell_type, env):
        """Draw a single cell of the environment"""
        rect_x = self.grid_x + x * self.cell_size
        rect_y = self.grid_y + y * self.cell_size
        
        # Get cell color based on visualization mode
        cell_color = self.get_cell_color_by_mode(x, y, cell_type, env)
        
        # Draw cell background
        pygame.draw.rect(self.screen, cell_color, (rect_x, rect_y, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, self.BLACK, (rect_x, rect_y, self.cell_size, self.cell_size), 1)
        
        # Draw room boundaries if in normal mode
        if self.visualization_mode == self.MODE_NORMAL and cell_type == env.EMPTY:
            # Check if this cell is at a room boundary
            x, y = int(x), int(y)
            if x > 0 and env.room_map[y, x] != env.room_map[y, x-1] and env.knowledge_map[y, x-1] == env.EMPTY:
                line_x = rect_x
                line_y = rect_y
                pygame.draw.line(self.screen, self.BLACK, (line_x, line_y), (line_x, line_y + self.cell_size), 3)
            
            if y > 0 and env.room_map[y, x] != env.room_map[y-1, x] and env.knowledge_map[y-1, x] == env.EMPTY:
                line_x = rect_x
                line_y = rect_y
                pygame.draw.line(self.screen, self.BLACK, (line_x, line_y), (line_x + self.cell_size, line_y), 3)
    
    def draw_robot(self, env):
        """Draw the robot with animation"""
        x, y = env.robot_pos
        
        # Calculate drawing position with animation offset
        if env.last_action is not None and env.animation_progress < 1.0:
            prev_x = x - env.ACTIONS[env.last_action][0]
            prev_y = y - env.ACTIONS[env.last_action][1]
            
            # Only animate if the previous position was valid
            if 0 <= prev_x < env.width and 0 <= prev_y < env.height:
                # Interpolate between previous and current position
                interp_x = prev_x + env.animation_progress * env.ACTIONS[env.last_action][0]
                interp_y = prev_y + env.animation_progress * env.ACTIONS[env.last_action][1]
                
                x, y = interp_x, interp_y
        
        # Draw robot image
        robot_img = self.robot_images[env.facing]
        img_rect = robot_img.get_rect()
        img_rect.center = (self.grid_x + (x + 0.5) * self.cell_size, 
                          self.grid_y + (y + 0.5) * self.cell_size)
        self.screen.blit(robot_img, img_rect)
    
    def draw_path(self, env):
        """Draw the robot's path history"""
        if not self.show_path or len(env.path_history) < 2:
            return
        
        # Draw path lines
        points = [(self.grid_x + (x + 0.5) * self.cell_size, 
                 self.grid_y + (y + 0.5) * self.cell_size) 
                 for x, y in env.path_history]
        
        # Use a semi-transparent surface for the path
        path_surface = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
        pygame.draw.lines(path_surface, (0, 255, 0, 128), False, points, 2)
        self.screen.blit(path_surface, (0, 0))
    
    def draw_optimal_path(self, env):
        """Draw the planned optimal path"""
        if not env.optimal_path or len(env.optimal_path) < 2:
            return
        
        # Draw planned path
        points = [(self.grid_x + (x + 0.5) * self.cell_size, 
                 self.grid_y + (y + 0.5) * self.cell_size) 
                 for x, y in env.optimal_path]
        
        # Use dashed line for optimal path
        path_surface = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
        
        # Draw dashed line manually
        for i in range(len(points) - 1):
            # Use red color with transparency
            pygame.draw.line(path_surface, (255, 0, 0, 180), points[i], points[i+1], 2)
        
        # Draw target point
        if env.target_position:
            tx, ty = env.target_position
            target_pos = (self.grid_x + (tx + 0.5) * self.cell_size, 
                         self.grid_y + (ty + 0.5) * self.cell_size)
            pygame.draw.circle(path_surface, (255, 0, 0, 200), target_pos, 8, 2)
        
        self.screen.blit(path_surface, (0, 0))
    
    def draw_stats(self, env, episode=1, steps=0, avg_reward=0):
        """Draw statistics and info text"""
        # Draw coverage text
        coverage_text = f"Coverage: {env.coverage*100:.1f}%"
        text_surface = self.font.render(coverage_text, True, self.BLACK)
        self.screen.blit(text_surface, (self.grid_x, 20))
        
        # Draw episode info
        episode_text = f"Episode: {episode}  Steps: {steps}"
        text_surface = self.font.render(episode_text, True, self.BLACK)
        self.screen.blit(text_surface, (self.grid_x, 50))
        
        # Draw reward info
        reward_text = f"Avg Reward: {avg_reward:.3f}"
        text_surface = self.font.render(reward_text, True, self.BLACK)
        self.screen.blit(text_surface, (self.grid_x + 300, 50))
        
        # Draw mode button
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, self.button_rect)
        pygame.draw.rect(self.screen, self.BLACK, self.button_rect, 1)  # Border
        
        # Button text depends on current mode
        mode_names = ["Room View", "Traffic Map", "Obstacle Map", "Discovery Map"]
        button_text = mode_names[self.visualization_mode]
        text_surface = self.button_font.render(button_text, True, self.WHITE)
        text_rect = text_surface.get_rect(center=self.button_rect.center)
        self.screen.blit(text_surface, text_rect)
    
    def draw_legend(self, env):
        """Draw a legend explaining the current visualization mode"""
        # Draw legend background
        pygame.draw.rect(self.screen, (240, 240, 240), self.legend_rect)
        pygame.draw.rect(self.screen, self.BLACK, self.legend_rect, 1)  # Border
        
        # Legend title based on visualization mode
        titles = ["Room Legend", "Traffic Legend", "Obstacle Legend", "Discovery Legend"]
        title_text = titles[self.visualization_mode]
        text_surface = self.small_font.render(title_text, True, self.BLACK)
        self.screen.blit(text_surface, (self.legend_rect.x + 10, self.legend_rect.y + 10))
        
        # Draw legend content
        y_offset = 30
        if self.visualization_mode == self.MODE_NORMAL:
            # Room colors
            for i in range(min(4, len(self.room_colors))):
                pygame.draw.rect(self.screen, self.room_colors[i], 
                               (self.legend_rect.x + 10, self.legend_rect.y + y_offset, 20, 20))
                text = f"Room {i+1}"
                text_surface = self.small_font.render(text, True, self.BLACK)
                self.screen.blit(text_surface, (self.legend_rect.x + 40, self.legend_rect.y + y_offset))
                y_offset += 25
        
        elif self.visualization_mode == self.MODE_TRAFFIC:
            # Traffic gradient
            colors = [(50, 255, 50), (255, 255, 50), (255, 50, 50)]
            labels = ["Low Traffic", "Medium", "High Traffic"]
            for i in range(3):
                pygame.draw.rect(self.screen, colors[i], 
                               (self.legend_rect.x + 10, self.legend_rect.y + y_offset, 20, 20))
                text_surface = self.small_font.render(labels[i], True, self.BLACK)
                self.screen.blit(text_surface, (self.legend_rect.x + 40, self.legend_rect.y + y_offset))
                y_offset += 25
        
        elif self.visualization_mode == self.MODE_OBSTACLE_DENSITY:
            # Obstacle density gradient
            colors = [(50, 255, 50), (255, 255, 50), (255, 50, 50)]
            labels = ["Few Obstacles", "Medium", "Many Obstacles"]
            for i in range(3):
                pygame.draw.rect(self.screen, colors[i], 
                               (self.legend_rect.x + 10, self.legend_rect.y + y_offset, 20, 20))
                text_surface = self.small_font.render(labels[i], True, self.BLACK)
                self.screen.blit(text_surface, (self.legend_rect.x + 40, self.legend_rect.y + y_offset))
                y_offset += 25
        
        elif self.visualization_mode == self.MODE_TIME_EFFICIENCY:
            # Discovery time gradient
            colors = [(50, 255, 100), (255, 255, 100), (255, 50, 100)]
            labels = ["Early Discovery", "Medium", "Late Discovery"]
            for i in range(3):
                pygame.draw.rect(self.screen, colors[i], 
                               (self.legend_rect.x + 10, self.legend_rect.y + y_offset, 20, 20))
                text_surface = self.small_font.render(labels[i], True, self.BLACK)
                self.screen.blit(text_surface, (self.legend_rect.x + 40, self.legend_rect.y + y_offset))
                y_offset += 25
    
    def render(self, env, episode=1, steps=0, avg_reward=0):
        """Render the current state of the environment"""
        # Handle animation
        if env.animation_progress < 1.0:
            env.animation_progress += 0.1  # Adjust for desired animation speed
        
        # Fill background
        self.screen.fill((230, 230, 230))
        
        # Draw cells
        state_map = env.get_state()['visual_state']
        for y in range(env.height):
            for x in range(env.width):
                cell_type = state_map[y, x]
                if cell_type != env.ROBOT:  # Don't draw robot here
                    self.draw_cell(x, y, cell_type, env)
        
        # Draw path first (so it's under the robot)
        self.draw_path(env)
        
        # Draw optimal path if it exists
        self.draw_optimal_path(env)
        
        # Draw robot
        self.draw_robot(env)
        
        # Draw stats
        self.draw_stats(env, episode, steps, avg_reward)
        
        # Draw legend
        self.draw_legend(env)
        
        # Update display
        pygame.display.flip()
    
    def close(self):
        """Clean up Pygame"""
        pygame.quit()


def create_simulator(render=True, episodes=3):
    """
    Create a robot vacuum simulator with Q-learning agent
    
    Args:
        render: Whether to render with PyGame
        episodes: Number of episodes to run
    """
    # Environment setup
    env = RobotVacuumEnvironment(width=15, height=12, obstacle_prob=0.2)
    agent = QLearningAgent(action_space=4, learning_rate=0.1, exploration_rate=0.5)
    
    # Visualization setup
    visualizer = PyGameVisualizer(cell_size=40) if render else None
    if visualizer:
        visualizer.setup(env)
    
    # Training loop
    total_reward = 0
    step_count = 0  # Total step counter
    episode_rewards = []  # Track rewards per episode
    
    for episode in range(1, episodes+1):
        # Reset environment for new episode
        if episode > 1:
            env = RobotVacuumEnvironment(width=15, height=12, obstacle_prob=0.2)
            # Keep agent's learning
        
        state = env.get_state()
        episode_reward = 0
        steps = 0
        done = False
        
        # Store starting path
        visualizer.path = [env.robot_pos] if visualizer else []
        
        while not done:
            # Choose action using Q-learning (sometimes with path optimization)
            if random.random() < 0.3:  # 30% chance to use planned path
                action = env.get_next_optimal_action()
            else:
                action = agent.get_action(state)
            
            # Take action and get next state
            next_state, reward, done = env.step(action)
            
            # Update Q-values
            agent.update_q_value(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            total_reward += reward
            step_count += 1
            
            # Visualization
            if visualizer:
                # Calculate average reward
                avg_reward = total_reward / step_count if step_count > 0 else 0
                
                event_result = visualizer.handle_events(env)
                if event_result == "new_episode":
                    # User requested new episode
                    break
                
                visualizer.render(env, episode, steps, avg_reward)
                
                # Limit speed
                time.sleep(0.05)
            
            # Episode length limit
            if steps >= 1000:
                done = True
        
        # Store episode reward
        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode}: {steps} steps, reward: {episode_reward:.2f}, coverage: {env.coverage*100:.1f}%")
        
    if visualizer:
        visualizer.close()
    
    return agent, episode_rewards


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Robot Vacuum Simulator')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    
    args = parser.parse_args()
    
    # Run simulator
    agent, rewards = create_simulator(render=not args.no_render, episodes=args.episodes)
    
    # Print final stats
    print("\nTraining completed!")
    print(f"Average reward per episode: {sum(rewards)/len(rewards):.2f}")
    
    # Save trained agent if needed
    # import pickle
    # with open('trained_vacuum_agent.pkl', 'wb') as f:
    #     pickle.dump(agent, f)