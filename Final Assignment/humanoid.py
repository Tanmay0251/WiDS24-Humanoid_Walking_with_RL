import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, revoluteJointDef)
import pygame
import math

def map_image_to_rect(image, vertices, screen):
    """
    Map an image to a rectangle defined by its vertices.
    """
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    center_x = sum(x_coords) / 4
    center_y = sum(y_coords) / 4
    width = math.dist(vertices[0], vertices[1])
    height = math.dist(vertices[0], vertices[3])
    dx = vertices[1][0] - vertices[0][0]
    dy = vertices[1][1] - vertices[0][1]
    angle = math.atan2(dy, dx)
    rotated_image = pygame.transform.rotate(image, -angle * 180 / math.pi)
    rotated_rect = rotated_image.get_rect(center=(center_x, center_y))
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

        # Define collision categories for body parts
        self.category_torso = 0x0001  # Torso category
        self.category_left_leg = 0x0002  # Left leg category
        self.category_right_leg = 0x0004  # Right leg category

        # Define mask for collision categories (what the body can collide with)
        self.mask_ground_walls = 0x0001  # Collide with ground and walls
        self.mask_no_self = 0xFFFF ^ (self.category_left_leg | self.category_right_leg)  # Avoid colliding with own parts

        # Create torso with collision filtering
        self.torso = self.world.CreateDynamicBody(position=(x, y),
                                                  fixtures=Box2D.b2FixtureDef(
                                                      shape=polygonShape(box=(self.torso_width / 2, self.torso_height / 2)),
                                                      density=1.0,
                                                      filter=Box2D.b2Filter(categoryBits=self.category_torso, maskBits=self.mask_ground_walls)))

        # Load torso image
        self.torso.image = pygame.image.load('./assets/torso.png')
        self.torso.image = pygame.transform.scale(self.torso.image, (int(self.torso_width * 100), int(self.torso_height * 100)))

        # Create left thigh with collision filtering
        self.left_thigh = self.world.CreateDynamicBody(position=(x - 0.15, y - self.torso_height / 2 - self.thigh_length / 2),
                                                       fixtures=Box2D.b2FixtureDef(
                                                           shape=polygonShape(box=(self.leg_width / 2, self.thigh_length / 2)),
                                                           density=1.0,
                                                           filter=Box2D.b2Filter(categoryBits=self.category_left_leg, maskBits=self.mask_ground_walls)))

        # Load left thigh image
        self.left_thigh.image = pygame.image.load('./assets/left_thigh.png')
        self.left_thigh.image = pygame.transform.scale(self.left_thigh.image, (int(self.leg_width * 100), int(self.thigh_length * 100)))

        # Create left shin with collision filtering
        self.left_shin = self.world.CreateDynamicBody(position=(x - 0.15, y - self.torso_height / 2 - self.thigh_length - self.shin_length / 2),
                                                    fixtures=Box2D.b2FixtureDef(
                                                        shape=polygonShape(box=(self.leg_width / 2, self.shin_length / 2)),
                                                        density=1.0,
                                                        filter=Box2D.b2Filter(categoryBits=self.category_left_leg, maskBits=self.mask_ground_walls)))

        # Load left shin image
        self.left_shin.image = pygame.image.load('./assets/left_shin.png')
        self.left_shin.image = pygame.transform.scale(self.left_shin.image, (int(self.leg_width * 100), int(self.shin_length * 100)))

        # Create right thigh with collision filtering
        self.right_thigh = self.world.CreateDynamicBody(position=(x + 0.15, y - self.torso_height / 2 - self.thigh_length / 2),
                                                        fixtures=Box2D.b2FixtureDef(
                                                            shape=polygonShape(box=(self.leg_width / 2, self.thigh_length / 2)),
                                                            density=1.0,
                                                            filter=Box2D.b2Filter(categoryBits=self.category_right_leg, maskBits=self.mask_ground_walls)))

        # Load right thigh image
        self.right_thigh.image = pygame.image.load('./assets/right_thigh.png')
        self.right_thigh.image = pygame.transform.scale(self.right_thigh.image, (int(self.leg_width * 100), int(self.thigh_length * 100)))

        # Create right shin with collision filtering
        self.right_shin = self.world.CreateDynamicBody(position=(x + 0.15, y - self.torso_height / 2 - self.thigh_length - self.shin_length / 2),
                                                    fixtures=Box2D.b2FixtureDef(
                                                        shape=polygonShape(box=(self.leg_width / 2, self.shin_length / 2)),
                                                        density=1.0,
                                                        filter=Box2D.b2Filter(categoryBits=self.category_right_leg, maskBits=self.mask_ground_walls)))

        # Load right shin image
        self.right_shin.image = pygame.image.load('./assets/right_shin.png')
        self.right_shin.image = pygame.transform.scale(self.right_shin.image, (int(self.leg_width * 100), int(self.shin_length * 100)))

        # Create joints (same as before)
        self.joints = []
        self.create_joints()

    def create_joints(self):
        """Create the joints between the body parts."""
        # Left hip joint
        joint = revoluteJointDef(bodyA=self.torso,
                                 bodyB=self.left_thigh,
                                 localAnchorA=(-0.15, -self.torso_height / 2),
                                 localAnchorB=(0, self.thigh_length / 2),
                                 enableMotor=True,
                                 maxMotorTorque=40.0)
        self.joints.append(self.world.CreateJoint(joint))

        # Left knee joint
        joint = revoluteJointDef(bodyA=self.left_thigh,
                                 bodyB=self.left_shin,
                                 localAnchorA=(0, -self.thigh_length / 2),
                                 localAnchorB=(0, self.shin_length / 2),
                                 enableMotor=True,
                                 maxMotorTorque=40.0)
        self.joints.append(self.world.CreateJoint(joint))

        # Right hip joint
        joint = revoluteJointDef(bodyA=self.torso,
                                 bodyB=self.right_thigh,
                                 localAnchorA=(0.15, -self.torso_height / 2),
                                 localAnchorB=(0, self.thigh_length / 2),
                                 enableMotor=True,
                                 maxMotorTorque=40.0)
        self.joints.append(self.world.CreateJoint(joint))

        # Right knee joint
        joint = revoluteJointDef(bodyA=self.right_thigh,
                                 bodyB=self.right_shin,
                                 localAnchorA=(0, -self.thigh_length / 2),
                                 localAnchorB=(0, self.shin_length / 2),
                                 enableMotor=True,
                                 maxMotorTorque=40.0)
        self.joints.append(self.world.CreateJoint(joint))

    def update_motors(self, motor_speeds):
        """
        Update motor speeds for each joint.
        """
        for i, speed in enumerate(motor_speeds):
            self.joints[i].motorSpeed = float(speed)

    def render(self, screen, ppm):
        """Render the humanoid on the screen."""
        for body in [self.torso, self.left_thigh, self.left_shin, self.right_thigh, self.right_shin]:
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = [(body.transform * v) * ppm for v in shape.vertices]
                vertices = [(v[0], 600 - v[1]) for v in vertices]  # Flip y-axis for rendering
                map_image_to_rect(body.image, vertices, screen)

    def log_state(self):
        """
        Logs the relevant state information.
        Returns:
            dict: Contains the state values (joint angles, positions, velocities, etc.).
        """
        state = {}
        for i, joint in enumerate(self.joints):
            joint_angle = joint.angle
            joint_velocity = joint.motorSpeed
            state[f'joint_{i}_angle'] = joint_angle
            state[f'joint_{i}_velocity'] = joint_velocity

        # Position and velocities of the torso and limbs
        state['torso_x'] = self.torso.position.x
        state['torso_y'] = self.torso.position.y
        state['torso_vx'] = self.torso.linearVelocity.x
        state['torso_vy'] = self.torso.linearVelocity.y

        # Left and right thighs, shins positions and velocities
        state['left_thigh_x'] = self.left_thigh.position.x
        state['left_thigh_y'] = self.left_thigh.position.y
        state['left_thigh_vx'] = self.left_thigh.linearVelocity.x
        state['left_thigh_vy'] = self.left_thigh.linearVelocity.y

        state['right_thigh_x'] = self.right_thigh.position.x
        state['right_thigh_y'] = self.right_thigh.position.y
        state['right_thigh_vx'] = self.right_thigh.linearVelocity.x
        state['right_thigh_vy'] = self.right_thigh.linearVelocity.y

        # Left and right shins positions and velocities
        state['left_shin_x'] = self.left_shin.position.x
        state['left_shin_y'] = self.left_shin.position.y
        state['left_shin_vx'] = self.left_shin.linearVelocity.x
        state['left_shin_vy'] = self.left_shin.linearVelocity.y

        state['right_shin_x'] = self.right_shin.position.x
        state['right_shin_y'] = self.right_shin.position.y
        state['right_shin_vx'] = self.right_shin.linearVelocity.x
        state['right_shin_vy'] = self.right_shin.linearVelocity.y

        return state
