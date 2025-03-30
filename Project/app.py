import pygame
import random
import numpy as np
import time

class RoombaEnv:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Roomba Simulation")

        self.font = pygame.font.Font(None, 36)
        self.roomba_radius = 20
        self.roomba_pos = np.array([width // 2, height // 2], dtype=float)
        self.roomba_speed = 3
        
        self.battery_level = 100
        self.steps_taken = 0
        self.steps_per_battery_decrease = 50
        self.battery_decrease_amount = 2
        self.charging_rate = 10
        self.last_charge_time = time.time()
        
        self.charging_station_pos = np.array([50, 50])
        self.charging_station_size = 40
        
        self.walls = []
        self.generate_walls()
        
        self.dirt_particles = []
        self.dirt_radius = 3
        self.generate_dirt(100)
        
        self.score = 0
        self.points_per_dirt = 10

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BROWN = (139, 69, 19)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)

        self.clock = pygame.time.Clock()
        self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

    def generate_walls(self):
        wall_width = 20
        self.walls = [
            pygame.Rect(0, 0, self.width, wall_width),
            pygame.Rect(0, self.height - wall_width, self.width, wall_width),
            pygame.Rect(0, 0, wall_width, self.height),
            pygame.Rect(self.width - wall_width, 0, wall_width, self.height),
            pygame.Rect(200, 150, wall_width, 300),
            pygame.Rect(400, 200, 200, wall_width),
            pygame.Rect(600, 400, wall_width, 200),
        ]

    def is_at_charging_station(self):
        return np.linalg.norm(self.roomba_pos - self.charging_station_pos) < (self.roomba_radius + self.charging_station_size / 2)

    def update_battery(self):
        if self.steps_taken >= self.steps_per_battery_decrease:
            self.battery_level = max(0, self.battery_level - self.battery_decrease_amount)
            self.steps_taken = 0
        if self.is_at_charging_station():
            self.battery_level = min(100, self.battery_level + self.charging_rate)

    def generate_dirt(self, num_particles):
        self.dirt_particles = []
        for _ in range(num_particles):
            while True:
                x = random.randint(self.dirt_radius, self.width - self.dirt_radius)
                y = random.randint(self.dirt_radius, self.height - self.dirt_radius)
                dirt_rect = pygame.Rect(x - self.dirt_radius, y - self.dirt_radius, self.dirt_radius * 2, self.dirt_radius * 2)
                if not any(wall.colliderect(dirt_rect) for wall in self.walls):
                    self.dirt_particles.append([x, y])
                    break

    def check_collision(self, new_pos):
        roomba_rect = pygame.Rect(new_pos[0] - self.roomba_radius, new_pos[1] - self.roomba_radius, self.roomba_radius * 2, self.roomba_radius * 2)
        return any(wall.colliderect(roomba_rect) for wall in self.walls)

    def clean_dirt(self):
        roomba_rect = pygame.Rect(self.roomba_pos[0] - self.roomba_radius, self.roomba_pos[1] - self.roomba_radius, self.roomba_radius * 2, self.roomba_radius * 2)
        self.dirt_particles = [dirt for dirt in self.dirt_particles if not roomba_rect.collidepoint(dirt[0], dirt[1])]
        self.score += self.points_per_dirt

    def move_roomba(self):
        if self.battery_level <= 10:
            direction = np.sign(self.charging_station_pos - self.roomba_pos)
        else:
            if random.random() < 0.1:
                self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            direction = self.direction

        new_pos = self.roomba_pos + np.array(direction) * self.roomba_speed
        if not self.check_collision(new_pos):
            self.roomba_pos = new_pos
            self.steps_taken += 1

        self.clean_dirt()

    def render(self):
        self.screen.fill(self.WHITE)
        pygame.draw.rect(self.screen, self.YELLOW, (self.charging_station_pos[0] - self.charging_station_size / 2, self.charging_station_pos[1] - self.charging_station_size / 2, self.charging_station_size, self.charging_station_size))
        
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.GRAY, wall)
        
        for dirt in self.dirt_particles:
            pygame.draw.circle(self.screen, self.BROWN, (int(dirt[0]), int(dirt[1])), self.dirt_radius)
        
        color = self.BLACK if self.battery_level > 50 else self.YELLOW if self.battery_level > 20 else self.RED
        pygame.draw.circle(self.screen, color, (int(self.roomba_pos[0]), int(self.roomba_pos[1])), self.roomba_radius)
        
        self.screen.blit(self.font.render(f"Score: {self.score}", True, self.BLACK), (10, 10))
        self.screen.blit(self.font.render(f"Battery: {int(self.battery_level)}%", True, self.BLACK), (10, 50))
        self.screen.blit(self.font.render(f"Dirt Remaining: {len(self.dirt_particles)}", True, self.BLACK), (10, 90))
        if self.is_at_charging_station():
            self.screen.blit(self.font.render("CHARGING", True, self.GREEN), (10, 130))
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.move_roomba()
            self.update_battery()
            self.render()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    env = RoombaEnv()
    env.run()