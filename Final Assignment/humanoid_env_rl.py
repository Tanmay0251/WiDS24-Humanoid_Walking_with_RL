import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from simulation import Simulation

class HumanoidEnv(gym.Env):
    """
    Custom OpenAI Gym environment for controlling a humanoid's walk.
    """

    def __init__(self):
        super(HumanoidEnv, self).__init__()

        # Initialize the simulation
        self.simulation = Simulation()
        self.simulation_clock = pygame.time.Clock()

        # Define the action space: motor speeds for 4 joints
        self.action_space = spaces.Box(
            low=np.array([-10, -10, -10, -10]),  # Min motor speeds
            high=np.array([10, 10, 10, 10]),    # Max motor speeds
            dtype=np.float32
        )

        # Define the observation space based on the humanoid's log_state
        self.observation_space = spaces.Box(
            low=-np.inf,  # Allow for unlimited observation values (customize if needed)
            high=np.inf,
            shape=(len(self.simulation.humanoid.log_state()),),
            dtype=np.float32
        )

    def reset(self,seed=None,   options =None):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        del self.simulation  # Clear the previous simulation
        self.simulation = Simulation()  # Restart the simulation
        return self._get_observation(), {}  

    def step(self, action):
        """
        Perform one step in the environment with the given action.

        Args:
            action (np.array): Motor speeds for the joints.

        Returns:
            observation (np.array): The new observation.
            reward (float): The computed reward.
            done (bool): Whether the episode is finished.
            info (dict): Additional debug information.
        """
        # Apply the action to the humanoid's motors
        self.simulation.humanoid.update_motors(action)

        # Step the simulation forward
        self.simulation.world.Step(1.0 / 60.0, 6, 2)

        # Get the new observation
        observation = self._get_observation()

        # Compute the reward
        reward = self._compute_reward()

        # Check if the episode is done
        done = self._is_done()

        # Additional info (can include debug data if needed)
        info = {}

        return observation, reward, done,False, info




    def render(self, mode='human'):
        """
        Render the environment using the simulation's rendering system.
        """
        self.simulation.screen.fill(self.simulation.bg_color)
        self.simulation.render_ground()
        self.simulation.render_flag()
        self.simulation.humanoid.render(self.simulation.screen, self.simulation.ppm)
        pygame.display.flip()
        self.simulation_clock.tick(600)

    def close(self):
        """
        Close the environment.
        """
        pygame.quit()

    def _get_observation(self):
        """
        Get the current observation from the humanoid's state.

        Returns:
            np.array: The state as a flattened array.
        """
        state = self.simulation.humanoid.log_state()
        return np.array(list(state.values()), dtype=np.float32)

    def _compute_reward(self):
        """
        Compute the reward based on the current state.

        Complete this function, more creativity gets more points

        You are also free to change model.py to change training time you will only be judged on the performance of the model when python3 model.py load is run


        Returns:
            float: The reward value.
        """
        ## complete this

        def pos(val):
            return val if val>0 else -1


        state = self.simulation.humanoid.log_state()

        x = state.get('torso_x', 0) # you can access these variable like this
        y = state.get('torso_y', 0)

        straight = 5000 #train these hyperparmaters depending on how the trained model performs
        frwd = 10
        frwd_v = 100
        done = 10000


        reward = (y-1.7)*straight + (x-2)*frwd + (pos(state.get('left_thigh_vx')) + pos(state.get('right_thigh_vx')) + pos(state.get('right_shin_vx')) + pos(state.get('left_shin_vx')) + state.get('torso_vx')) * frwd_v

        if reward>14.5:
            reward+=done


        return reward

    def _is_done(self):
        """
        Determine if the episode is done.

        Returns:
            bool: True if the episode is finished, otherwise False.
        """
        state = self.simulation.humanoid.log_state()
        x = state.get('torso_x', 0)

        # End the episode if the humanoid falls
        if x >14.5:
            return True

        return False
