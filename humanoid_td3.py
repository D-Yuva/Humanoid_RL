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

# Hyperparameters (Optimized for TD3 and standing task)
EPISODES = 10000000  # Total episodes
STEPS_PER_EPISODE = 200  # Steps per episode
LEARNING_RATE_ACTOR = 0.0003  # Actor learning rate
LEARNING_RATE_CRITIC = 0.0003  # Critic learning rate
DISCOUNT_FACTOR = 0.99  # Discount factor
BATCH_SIZE = 256  # Batch size for training
BUFFER_SIZE = 100000  # Replay buffer size
TAU = 0.005  # Target network update rate
POLICY_NOISE = 0.2  # Noise added to target policy
NOISE_CLIP = 0.5  # Noise clip for policy noise
POLICY_DELAY = 2  # Delay for policy updates
EXPLORATION_NOISE = 0.1  # Exploration noise
HIDDEN_LAYER_SIZE = 256  # Network capacity
NUM_SERVOS = 17  # Number of servo motors in robot
REWARD_SCALE = 1.0  # Scale reward for faster learning
MODEL_SAVE_INTERVAL = 100  # Save model every 100 episodes
MODEL_SAVE_PATH = "humanoid_td3_model.pth"  # Model save path


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


# Replay Buffer for TD3
class ReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )


# Actor Network for TD3
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh()  # Output between -1 and 1
        )
        
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.max_action * self.network(x)


# Critic Network for TD3 (Twin Critics)
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for layer in self.q1:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
        
        for layer in self.q2:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x)


# TD3 Agent
class TD3:
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        max_action=1.0
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.max_action = max_action
        
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize actor and critics
        self.actor = Actor(self.obs_dim, self.act_dim, hidden_size, max_action).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.act_dim, hidden_size, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = Critic(self.obs_dim, self.act_dim, hidden_size).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.act_dim, hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, self.obs_dim, self.act_dim)
        
        # Training variables
        self.training_steps = 0

    def select_action(self, state, add_noise=True):
        """Select action based on current policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state).cpu().numpy().flatten()
            
            if add_noise:
                noise = np.random.normal(0, EXPLORATION_NOISE, size=self.act_dim)
                action = np.clip(action + noise, -self.max_action, self.max_action)
                
        return action

    def train(self):
        """Training step for TD3"""
        # Sample from replay buffer
        if self.replay_buffer.size < BATCH_SIZE:
            return
            
        self.training_steps += 1
        
        state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
        
        # Update critics
        with torch.no_grad():
            # Select action according to target policy and add noise for smoothing
            noise = torch.randn_like(action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_state) + noise
            next_action = torch.clamp(next_action, -self.max_action, self.max_action)
            
            # Compute target Q values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.training_steps % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filename}")


if __name__ == '__main__':
    env = HumanoidStandEnv(render=True)  # Set render=True for visualization
    observation_space = env.observation_space
    action_space = env.action_space

    agent = TD3(
        observation_space, 
        action_space, 
        hidden_size=HIDDEN_LAYER_SIZE, 
        lr_actor=LEARNING_RATE_ACTOR,
        lr_critic=LEARNING_RATE_CRITIC,
        gamma=DISCOUNT_FACTOR,
        tau=TAU,
        policy_noise=POLICY_NOISE,
        noise_clip=NOISE_CLIP,
        policy_delay=POLICY_DELAY,
        max_action=1.0
    )

    # Training loop
    try:
        total_timesteps = 0
        for episode in range(EPISODES):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            for step in range(env.max_episode_steps):
                # Select action with exploration noise
                action = agent.select_action(state)
                
                # Perform action
                next_state, reward, done, _, _ = env.step(action)
                
                # Store data in replay buffer
                agent.replay_buffer.add(state, action, next_state, reward, float(done))
                
                # Update state
                state = next_state
                episode_reward += reward
                total_timesteps += 1
                
                # Train agent
                agent.train()
                
                if done:
                    break
            
            print(f"Episode {episode+1}/{EPISODES} | Reward: {episode_reward:.2f} | Steps: {step+1}")
            
            # Save model periodically
            if (episode + 1) % MODEL_SAVE_INTERVAL == 0:
                agent.save(MODEL_SAVE_PATH)
                
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()