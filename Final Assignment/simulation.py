import pygame
import Box2D
from Box2D.b2 import world, polygonShape
from humanoid import Humanoid

class Simulation:
    def __init__(self):
        pygame.init()
        self.width, self.height = 1600, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("2D Humanoid Simulation")

        # Pixels per meter (scaling factor)
        self.ppm = 100

        # Colors
        self.bg_color = (255, 255, 255)
        self.ground_color = (0, 0, 0)

        # Load flag image and scale it
        self.flag_image = pygame.image.load('./assets/flag.png')
        self.flag_image = pygame.transform.scale(self.flag_image, (50, 50))  # Resize to make it small

        # Create the Box2D world
        self.world = world(gravity=(0, -10), doSleep=True)

        # Create ground
        self.ground_height = 50 / self.ppm
        self.ground = self.world.CreateStaticBody(
            position=(self.width / (2 * self.ppm), self.ground_height / 2),
            shapes=polygonShape(box=(self.width / (2 * self.ppm), self.ground_height / 2))
        )

        # Create left and right boundary walls
        self.left_wall = self.world.CreateStaticBody(
            position=(0, self.height / (2 * self.ppm)),
            shapes=polygonShape(box=(self.ground_height / 2, self.height / (2 * self.ppm)))
        )

        self.right_wall = self.world.CreateStaticBody(
            position=(self.width / self.ppm, self.height / (2 * self.ppm)),
            shapes=polygonShape(box=(self.ground_height / 2, self.height / (2 * self.ppm)))
        )

        # Create humanoid
        self.humanoid = Humanoid(self.world, x=(self.width-1200) / (2 * self.ppm), y=self.height / self.ppm - self.ground_height)

    def render_ground(self):
        """Render the ground."""
        pygame.draw.rect(
            self.screen,
            self.ground_color,
            pygame.Rect(0, self.height - int(self.ground_height * self.ppm), self.width, int(self.ground_height * self.ppm)),
        )

    def render_flag(self):
        """Render the flag image at the right of the window at ground level."""
        flag_x = self.width - self.flag_image.get_width() - 10  # 10px from the right edge
        flag_y = self.height - int(self.ground_height * self.ppm) - self.flag_image.get_height()
        self.screen.blit(self.flag_image, (flag_x, flag_y))

    def render_walls(self):
        """Render the boundary walls."""
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),  # Color for the walls (black)
            pygame.Rect(0, 0, int(self.ground_height * self.ppm), self.height),  # Left wall
        )
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),  # Color for the walls (black)
            pygame.Rect(self.width - int(self.ground_height * self.ppm), 0, int(self.ground_height * self.ppm), self.height),  # Right wall
        )

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Clear the screen
            self.screen.fill(self.bg_color)

            # Render ground
            self.render_ground()

            # Render walls
            self.render_walls()

            # Render flag
            self.render_flag()

            # Render humanoid
            self.humanoid.render(self.screen, self.ppm)

            self.humanoid.update_motors(motor_speeds=[0, 0, 0, 0])

            if self.humanoid.torso.position.x > 16:
                print("finish")
                break

            # Update Box2D world
            self.world.Step(1.0 / 60.0, 6, 2)

            # Update the display
            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()
