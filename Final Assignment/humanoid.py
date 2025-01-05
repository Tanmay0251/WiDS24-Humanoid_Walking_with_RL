    # humanoid.py
import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, revoluteJointDef)
import pygame
import math

def map_image_to_rect(image, vertices, screen):
    """
    Map an image to a rectangle defined by its vertices.
    
    :param image: The image to be mapped.
    :param vertices: The vertices of the rectangle, e.g., [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    :param screen: The screen to render the image on.
    """
    # Calculate center of the rectangle (mean of all vertices)
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    center_x = sum(x_coords) / 4
    center_y = sum(y_coords) / 4

    # Calculate width and height (distance between two opposite corners)
    width = math.dist(vertices[0], vertices[1])  # Distance between (x1, y1) and (x2, y2)
    height = math.dist(vertices[0], vertices[3])  # Distance between (x1, y1) and (x4, y4)
    
    # Calculate the angle of rotation (between the horizontal and the first side of the rectangle)
    dx = vertices[1][0] - vertices[0][0]
    dy = vertices[1][1] - vertices[0][1]
    angle = math.atan2(dy, dx)  # Angle between the x-axis and the first edge
    
    # Rotate the image
    rotated_image = pygame.transform.rotate(image, -angle * 180 / math.pi)  # Convert radians to degrees
    rotated_rect = rotated_image.get_rect(center=(center_x, center_y))

    # Draw the rotated image
    screen.blit(rotated_image, rotated_rect)

class Humanoid:
    def __init__(self, world, x, y):
        self.world = world

        # Body part dimensions (in meters, Box2D uses meters)
        self.thigh_length = 0.5
        self.shin_length = 0.5
        self.torso_height = 1.0
        self.torso_width = 0.3
        self.leg_width = 0.2

        # Create torso
        self.torso = self.world.CreateDynamicBody(position=(x, y),
                                                  fixtures=Box2D.b2FixtureDef(shape=polygonShape(box=(self.torso_width / 2, self.torso_height / 2)),
                                                                             density=1.0))
        # Load torso image
        self.torso.image = pygame.image.load('.assets/torso.png')
        self.torso.image = pygame.transform.scale(self.torso.image, (int(self.torso_width * 100), int(self.torso_height * 100)))

        # Create left thigh
        self.left_thigh = self.world.CreateDynamicBody(position=(x - 0.15, y - self.torso_height / 2 - self.thigh_length / 2),
                                                       fixtures=Box2D.b2FixtureDef(shape=polygonShape(box=(self.leg_width / 2, self.thigh_length / 2)),
                                                                                  density=1.0))
        # Load left thigh image
        self.left_thigh.image = pygame.image.load('.assets/left_thigh.png')
        self.left_thigh.image = pygame.transform.scale(self.left_thigh.image, (int(self.leg_width * 100), int(self.thigh_length * 100)))

        # Create left shin
        self.left_shin = self.world.CreateDynamicBody(position=(x - 0.15, y - self.torso_height / 2 - self.thigh_length - self.shin_length / 2),
                                                    fixtures=Box2D.b2FixtureDef(shape=polygonShape(box=(self.leg_width / 2, self.shin_length / 2)),
                                                                                density=1.0))
        # Load left shin image
        self.left_shin.image = pygame.image.load('.assets/left_shin.png')
        self.left_shin.image = pygame.transform.scale(self.left_shin.image, (int(self.leg_width * 100), int(self.shin_length * 100)))

        # Create right thigh
        self.right_thigh = self.world.CreateDynamicBody(position=(x + 0.15, y - self.torso_height / 2 - self.thigh_length / 2),
                                                        fixtures=Box2D.b2FixtureDef(shape=polygonShape(box=(self.leg_width / 2, self.thigh_length / 2)),
                                                                                density=1.0))
        # Load right thigh image
        self.right_thigh.image = pygame.image.load('.assets/right_thigh.png')
        self.right_thigh.image = pygame.transform.scale(self.right_thigh.image, (int(self.leg_width * 100), int(self.thigh_length * 100)))

        # Create right shin
        self.right_shin = self.world.CreateDynamicBody(position=(x + 0.15, y - self.torso_height / 2 - self.thigh_length - self.shin_length / 2),
                                                    fixtures=Box2D.b2FixtureDef(shape=polygonShape(box=(self.leg_width / 2, self.shin_length / 2)),
                                                                                density=1.0))
        # Load right shin image
        self.right_shin.image = pygame.image.load('.assets/right_shin.png')
        self.right_shin.image = pygame.transform.scale(self.right_shin.image, (int(self.leg_width * 100), int(self.shin_length * 100)))


        # Create joints
        self.joints = []

        # Left hip joint
        joint = revoluteJointDef(bodyA=self.torso,
                                 bodyB=self.left_thigh,
                                 localAnchorA=(-0.15, -self.torso_height / 2),
                                 localAnchorB=(0, self.thigh_length / 2),
                                 enableMotor=True,
                                 maxMotorTorque=100.0)
        self.joints.append(self.world.CreateJoint(joint))

        # Left knee joint
        joint = revoluteJointDef(bodyA=self.left_thigh,
                                 bodyB=self.left_shin,
                                 localAnchorA=(0, -self.thigh_length / 2),
                                 localAnchorB=(0, self.shin_length / 2),
                                 enableMotor=True,
                                 maxMotorTorque=100.0)
        self.joints.append(self.world.CreateJoint(joint))

        # Right hip joint
        joint = revoluteJointDef(bodyA=self.torso,
                                 bodyB=self.right_thigh,
                                 localAnchorA=(0.15, -self.torso_height / 2),
                                 localAnchorB=(0, self.thigh_length / 2),
                                 enableMotor=True,
                                 maxMotorTorque=100.0)
        self.joints.append(self.world.CreateJoint(joint))

        # Right knee joint
        joint = revoluteJointDef(bodyA=self.right_thigh,
                                 bodyB=self.right_shin,
                                 localAnchorA=(0, -self.thigh_length / 2),
                                 localAnchorB=(0, self.shin_length / 2),
                                 enableMotor=True,
                                 maxMotorTorque=100.0)
        self.joints.append(self.world.CreateJoint(joint))

    def update_motors(self, motor_speeds):
        """
        Update motor speeds.
        motor_speeds: List of speeds [hip_left, knee_left, hip_right, knee_right].
        use this function for the model to control the subject.
        """
        for i, speed in enumerate(motor_speeds):
            self.joints[i].motorSpeed = float(speed)

    def render(self, screen, ppm):
        """Render the humanoid on the screen."""
        """Render the humanoid on the screen."""

        for body in [self.torso, self.left_thigh, self.left_shin, self.right_thigh, self.right_shin]:
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = [(body.transform * v) * ppm for v in shape.vertices]
                vertices = [(v[0], 600 - v[1]) for v in vertices]  # Flip y-axis for rendering
                map_image_to_rect(body.image,vertices,screen)


    def log_state(self):
        """
        Logs the relevant state information for RL.
        Returns:
            dict: Contains the state values (joint angles, positions, velocities, etc.).
        """
        state = {}

        # Get positions and velocities of joints
        for i, joint in enumerate(self.joints):
            joint_angle = joint.angle
            joint_velocity = joint.motorSpeed

            # Store joint states (joint angle and velocity)
            state[f'joint_{i}_angle'] = joint_angle
            state[f'joint_{i}_velocity'] = joint_velocity

        # Position of the torso
        state['torso_x'] = self.torso.position.x
        state['torso_y'] = self.torso.position.y
        state['torso_vx'] = self.torso.linearVelocity.x
        state['torso_vy'] = self.torso.linearVelocity.y

        # Position and velocities for the legs
        state['left_thigh_x'] = self.left_thigh.position.x
        state['left_thigh_y'] = self.left_thigh.position.y
        state['left_thigh_vx'] = self.left_thigh.linearVelocity.x
        state['left_thigh_vy'] = self.left_thigh.linearVelocity.y

        state['right_thigh_x'] = self.right_thigh.position.x
        state['right_thigh_y'] = self.right_thigh.position.y
        state['right_thigh_vx'] = self.right_thigh.linearVelocity.x
        state['right_thigh_vy'] = self.right_thigh.linearVelocity.y

        # Add additional state information as needed
        return state
