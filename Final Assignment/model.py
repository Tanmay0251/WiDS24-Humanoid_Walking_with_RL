import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from humanoid_env_rl import HumanoidEnv
import sys

# Create the humanoid environment
env = HumanoidEnv()

# Check the environment for compatibility with OpenAI Gym standards
check_env(env, warn=True)

# Wrap the environment with a Monitor to log episode rewards and lengths
env = Monitor(env)

# Define the RL model
model = PPO(
    policy="MlpPolicy",  # Multilayer perceptron policy
    env=env,              # Pass the custom environment
    verbose=1,            # Print training information
    tensorboard_log="./humanoid_rl_tensorboard/",  # Log directory for TensorBoard
    learning_rate=3e-1,   # Learning rate for optimization
    gamma=0.99,           # Discount factor
    n_steps=2048,         # Number of steps to run per rollout
    batch_size=64,        # Minibatch size for training
    n_epochs=10,          # Number of optimization epochs per update
)

# # Train the model
if sys.argv[1]=="load":
    model = PPO.load("humanoid_ppo_model", env=env)
else:
    TIMESTEPS = 10000  # Set the number of timesteps for training
    model.learn(total_timesteps=TIMESTEPS)
    model.save("humanoid_ppo_model")

# Test the trained model
obs = env.reset()[0]
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)  # Get action from the model
    # print(env.step(action) ) # uncomment to see the output
    obs, reward, done,_, info = env.step(action)  # Perform the action in the environment
    env.render()  # Render the environment (visualization)

# Close the environment
env.close()
