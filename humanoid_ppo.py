import pybullet as p
import pybullet_data
import time
import math
import random
import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal  # Import Normal distribution

# Hyperparameters (Optimized for RTX 3060 and standing task)
EPISODES = 500  # Increased for better learning
STEPS_PER_EPISODE = 200  # Increased for stability task
LEARNING_RATE = 0.0001  # Adjusted learning rate (lower for PPO)
DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.95  # GAE parameter
CLIP_RATIO = 0.2  # PPO clip ratio
TARGET_KL = 0.01  # Target KL divergence
UPDATE_EPOCHS = 10 # reduced number of epochs
BATCH_SIZE = 64  # Batch size for training
VALUE_COEF = 0.5  # Value function loss coefficient
ENTROPY_COEF = 0.01  # Entropy bonus coefficient
MAX_GRAD_NORM = 0.5  # Gradient clipping norm
HIDDEN_LAYER_SIZE = 256  # Increased network capacity
NUM_SERVOS = 17  # Number of servo motors in your robot
REWARD_SCALE = 1.0  # Scale up rewards for faster learning
MODEL_SAVE_INTERVAL = 100  # Save model every 100 episodes
MODEL_SAVE_PATH = "humanoid_ppo_model.pth"  # Model save file name



class HumanoidStandEnv(gym.Env):
    def __init__(self, render=False):
        self.render = render
        self.physicsClient = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient)  # Set gravity
        self.timeStep = 1.0 / 240  # Reduced timestep for stability
        p.setTimeStep(self.timeStep, physicsClientId=self.physicsClient)
        self.max_episode_steps = STEPS_PER_EPISODE

        self.robotId = None
        self.joint_ids = []
        self.num_joints = 0
        self.torque = 1.0  # Reduced default torque
        self.max_motor_force = 10.0  # added max motor force
        self.target_velocities = [0.0] * NUM_SERVOS  # Added target velocities

        # Define action and observation space (after robot is loaded)
        self.action_space = None
        self.observation_space = None

        self.reset()

    def reset(self, *, seed=None, options=None):  # Added seed for reproducibility
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient)

        # Load the URDF from the absolute path to the directory containing the script
        urdf_path = os.path.join("urdf/humanoidV3.urdf") 
        self.robotId = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=False,
                                  flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.physicsClient)
        
        p.loadURDF("plane.urdf", physicsClientId=self.physicsClient) # Load ground plane

        self.joint_ids = []
        self.num_joints = p.getNumJoints(self.robotId, physicsClientId=self.physicsClient)
        for j in range(self.num_joints):
            joint_info = p.getJointInfo(self.robotId, j, physicsClientId=self.physicsClient)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_ids.append(j)
                p.setJointMotorControl2(self.robotId, j, controlMode=p.VELOCITY_CONTROL, force=0,
                                        physicsClientId=self.physicsClient)
                p.enableJointForceTorqueSensor(self.robotId, j, enableSensor=True,
                                               physicsClientId=self.physicsClient)

        # Define action and observation space here, after robot is loaded and joint_ids are populated
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.joint_ids),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(len(self.joint_ids) * 2 + 3,),
                                             dtype=np.float32)  # Joint angles, velocities, base orientation

        # Reset joint angles to a standing pose (example)
        initial_pose = [0.0] * len(self.joint_ids)
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.robotId, joint_id, initial_pose[i], 0, physicsClientId=self.physicsClient)

        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        joint_states = p.getJointStates(self.robotId, self.joint_ids, physicsClientId=self.physicsClient)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        # Get base orientation (quaternion)
        base_orientation_quat = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.physicsClient)[1]
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        base_orientation_euler = p.getEulerFromQuaternion(base_orientation_quat)

        observation = np.array(joint_positions + joint_velocities + list(base_orientation_euler),
                               dtype=np.float32)  # Concatenate all observations

        return observation

    def step(self, action):
        scaled_action = np.clip(action, -1, 1) * self.max_motor_force  # added action scaling
        # Apply action as target motor velocities
        for i, joint_id in enumerate(self.joint_ids):
            # Apply action as desired motor velocities with torque limits
            p.setJointMotorControl2(self.robotId, joint_id, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=scaled_action[i], force=self.torque, maxVelocity=10.0,
                                    physicsClientId=self.physicsClient)

        p.stepSimulation(physicsClientId=self.physicsClient)
        if self.render:
            time.sleep(self.timeStep)  # keep this for rendering mode

        observation = self._get_observation()
        reward, done = self._get_reward_and_done()
        info = {}

        return observation, reward, done, False, info  # 'truncated' is always False for now

    def _get_reward_and_done(self):
        # Get robot base orientation (roll, pitch)
        base_orientation_quat = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.physicsClient)[1]
        roll, pitch, _ = p.getEulerFromQuaternion(base_orientation_quat)

        # Reward for staying upright (negative pitch punishes falling)
        upright_reward = math.cos(roll) * math.cos(pitch)

        # Penalty for large joint velocities (discourage jerky movements)
        joint_states = p.getJointStates(self.robotId, self.joint_ids, physicsClientId=self.physicsClient)
        joint_velocities = [state[1] for state in joint_states]
        velocity_penalty = -np.mean(np.abs(joint_velocities)) * 0.01  # Small penalty

        # Check if fallen (pitch or roll past certain threshold)
        done = abs(pitch) > 1.0 or abs(roll) > 1.0

        reward = (upright_reward + velocity_penalty) * REWARD_SCALE  # Scale up the reward
        return reward, done

    def close(self):
        p.disconnect(physicsClientId=self.physicsClient)




# Actor Network with separate mean and std output
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),  # Added extra layer
            nn.Tanh()
        )
        # Mean and log_std layers
        self.mean_layer = nn.Linear(hidden_size // 2, act_dim)
        self.log_std_layer = nn.Linear(hidden_size // 2, act_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)

        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, -1.0)  # Start with smaller std

    def forward(self, x):
        features = self.network(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2) # Clamp log_std for stability
        std = torch.exp(log_std)
        return mean, std


# Critic Network
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),  # Added extra layer
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.network(x)



class PPO:
    def __init__(self, observation_space, action_space,
                 hidden_size=256,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_ratio=0.2,
                 target_kl=0.01,
                 update_epochs=10,
                 batch_size=64,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5):

        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm


        self.actor = Actor(self.obs_dim, self.act_dim, hidden_size)
        self.critic = Critic(self.obs_dim, hidden_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # corrected device usage
        self.actor.to(self.device)
        self.critic.to(self.device)


    def select_action(self, observation):
        """Select action based on current policy (stochastic)"""
        with torch.no_grad(): # crucial for not storing gradients during action selection
            observation = torch.FloatTensor(observation).to(self.device)
            mean, std = self.actor(observation)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1) # Sum log probs for each action dimension
            value = self.critic(observation).item()  # Get single value from critic

        self.states.append(observation.cpu().numpy())  # Store numpy array
        self.actions.append(action.cpu().numpy())  # Store numpy array
        self.log_probs.append(log_prob.cpu().item())  # Store single log prob value
        self.values.append(value)  # Store single value

        return action.cpu().numpy()

    def update(self, observation, action, reward, next_observation, done):
        self.rewards.append(reward)
        self.dones.append(done)

        if done:
            self._update_policy()
            self._clear_memory()

    def _compute_advantages(self):
        """Compute advantages using GAE"""
        values = np.array(self.values)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextvalue = 0
            else:
                nextvalue = values[t+1]
            delta = rewards[t] + self.gamma * nextvalue * (1-dones[t]) - values[t]
            advantages_t = delta + self.gamma * self.gae_lambda * (1-dones[t]) * lastgaelam
            lastgaelam = advantages_t
            advantages[t] = advantages_t
            returns[t] = advantages_t + values[t]
        return advantages, returns

    def _update_policy(self):
        """PPO update"""
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)

        advantages, returns = self._compute_advantages()
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            # Mini-batch PPO update
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_old_log_probs = old_log_probs[start:end]
                batch_advantages = advantages[start:end]
                batch_returns = returns[start:end]

                # Get new log probs and values for batch
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1) # Sum for each action dimension
                entropy = dist.entropy().mean()  # Calculate entropy
                values = self.critic(batch_states).squeeze()


                # PPO Loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = -(torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages)).mean()
                critic_loss = F.mse_loss(values, batch_returns)


                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy # include entropy bonus
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                # Gradient Clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Calculate approx_kl
                with torch.no_grad():
                    log_ratio = log_probs - batch_old_log_probs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    if approx_kl > self.target_kl:
                        return

    def _clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device) # Added map_location
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Model loaded from {filename}")




if __name__ == '__main__':
    env = HumanoidStandEnv(render=True)  # Corrected: Set render=False for training
    observation_space = env.observation_space
    action_space = env.action_space


    agent = PPO(observation_space, action_space, hidden_size=HIDDEN_LAYER_SIZE, lr=LEARNING_RATE,
                gamma=DISCOUNT_FACTOR, gae_lambda=GAE_LAMBDA, clip_ratio=CLIP_RATIO, target_kl=TARGET_KL,
                update_epochs=UPDATE_EPOCHS, batch_size=BATCH_SIZE, value_coef=VALUE_COEF,
                entropy_coef=ENTROPY_COEF, max_grad_norm=MAX_GRAD_NORM)

    try:
        for episode in range(EPISODES):
            state, _ = env.reset()
            total_reward = 0
            for step in range(env.max_episode_steps): # Changed to env.max_episode_steps
                action = agent.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

            print(f"Episode: {episode}, Reward: {total_reward:.2f}")

            if episode % MODEL_SAVE_INTERVAL == 0:
                agent.save(MODEL_SAVE_PATH)


    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()