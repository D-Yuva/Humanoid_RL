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
import matplotlib.pyplot as plt  # Import for plotting

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Hyperparameters (Optimized for TD3 and standing task)
EPISODES = 1000  # Total episodes
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
REWARD_PLOT_PATH = "reward_plot.png"  # Path to save reward plot
EPISODE_REWARD_PATH = "episode_rewards.npy"  # Path to save episode rewards
EPISODE_DURATION_PATH = "episode_durations.npy"  # Path to save episode durations

CONTINUE_TRAINING = False  # Whether to continue training from a saved model
MAX_STAND_DURATION = 10.0  # Maximum stand duration in seconds for scaling reward
VISUALIZE_EPISODE = 50  # Visualize every 50th episode

class HumanoidStandEnv(gym.Env):
    def __init__(self, render=False, num_envs=1):  # Added num_envs
        self.render = render
        self.num_envs = num_envs  # Number of parallel environments
        self.physicsClient = []
        self.use_gui = render  # Separate variable to control GUI
        if self.use_gui:  # Connect one GUI client
            self.physicsClient.append(p.connect(p.GUI))
            for _ in range(num_envs - 1): # Then, DIRECT clients
                self.physicsClient.append(p.connect(p.DIRECT)) #Other Envs Do Not Render

        else:
             self.physicsClient = [p.connect(p.DIRECT) for _ in range(num_envs)]

        for client in self.physicsClient:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
            p.setGravity(0, 0, -9.81, physicsClientId=client)  # Set gravity

        self.timeStep = 1.0 / 240  # Reduced timestep for stability

        for client in self.physicsClient:
            p.setTimeStep(self.timeStep, physicsClientId=client)

        self.max_episode_steps = STEPS_PER_EPISODE

        self.robotId = [None] * num_envs
        self.joint_ids = [[] for _ in range(num_envs)]
        self.num_joints = 0
        self.torque = 1.0  # Reduced default torque
        self.max_motor_force = 10.0  # added max motor force
        self.target_velocities = [0.0] * NUM_SERVOS  # Added target velocities

        # Define action and observation space (after robot is loaded)
        self.action_space = None
        self.observation_space = None
        self.sensor_indices = {  # Define sensor link names and their indices
            "sensor1": None,
            "sensor2": None,
            "sensor3": None,
            "sensor4": None,
            "sensor5": None,
            "sensor6": None,
            "sensor7": None
        }

        self.episode_start_time = [0.0] * num_envs  # Track episode start time for each env
        self.episode_durations = [0.0] * num_envs  # Track episode duration for each env

        self.reset()

    def reset(self, *, seed=None, options=None):  # Added seed for reproducibility
        super().reset(seed=seed)
        observations = []
        infos = []
        for env_id in range(self.num_envs):
            client = self.physicsClient[env_id]
            p.resetSimulation(physicsClientId=client)
            p.setGravity(0, 0, -9.81, physicsClientId=client)

            # Load the URDF from the absolute path to the directory containing the script
            urdf_path = os.path.join("urdf/humanoidV3.urdf")
            self.robotId[env_id] = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=False,
                                      flags=p.URDF_USE_SELF_COLLISION, physicsClientId=client)
            p.loadURDF("plane.urdf", physicsClientId=client)  # Load ground plane

            self.joint_ids[env_id] = []
            num_joints = p.getNumJoints(self.robotId[env_id], physicsClientId=client)
            for j in range(num_joints):
                joint_info = p.getJointInfo(self.robotId[env_id], j, physicsClientId=client)
                if joint_info[2] == p.JOINT_REVOLUTE:
                    self.joint_ids[env_id].append(j)
                    p.setJointMotorControl2(self.robotId[env_id], j, controlMode=p.VELOCITY_CONTROL, force=0,
                                            physicsClientId=client)
                    p.enableJointForceTorqueSensor(self.robotId[env_id], j, enableSensor=True,
                                                   physicsClientId=client)

            # Initialize sensor indices - CORRECTED SENSOR INDEXING
            if env_id == 0:  # Only populate sensor indices once
                for sensor_name in self.sensor_indices:
                    try:
                        for j in range(p.getNumJoints(self.robotId[env_id], physicsClientId=client)):
                            joint_info = p.getJointInfo(self.robotId[env_id], j, physicsClientId=client)
                            if joint_info[1].decode('utf-8') == sensor_name:
                                self.sensor_indices[sensor_name] = j  # CORRECT: Store joint *index*
                                break  # Break once the sensor is found
                    except Exception as e:
                        print(f"Error finding sensor {sensor_name}: {e}")


            # Define action and observation space here, after robot is loaded and joint_ids are populated
            if self.action_space is None:  # Only define once
                self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.joint_ids[env_id]),), dtype=np.float32)
            if self.observation_space is None:  # Only define once
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                     shape=(len(self.joint_ids[env_id]) * 2 + 3 + 6 * len(self.sensor_indices),),  # Joint angles, velocities, base orientation, sensor data
                                                     dtype=np.float32)

            # Reset joint angles to a standing pose (example)
            initial_pose = [0.0] * len(self.joint_ids[env_id])
            for i, joint_id in enumerate(self.joint_ids[env_id]):
                p.resetJointState(self.robotId[env_id], joint_id, initial_pose[i], 0, physicsClientId=client)

            observation = self._get_observation(env_id)
            observations.append(observation)
            infos.append({})  # Append an empty dictionary for info

            self.episode_start_time[env_id] = time.time()  # Record start time
            self.episode_durations[env_id] = 0.0  # Reset duration

        return np.array(observations, dtype=np.float32), infos

    def _get_sensor_readings(self, env_id):
        sensor_readings = []
        # print(f"sensor_indices = {self.sensor_indices}")
        for sensor_name, sensor_index in self.sensor_indices.items():
            # print(f"{sensor_name} = {sensor_index}")
            if sensor_index is not None:
                try:
                    # Get sensor data (linear acceleration and angular velocity)
                    sensor_state = p.getJointState(self.robotId[env_id], sensor_index, physicsClientId=self.physicsClient[env_id]) # Correct call
                    sensor_readings.extend(sensor_state[2:5]) # angular velocity # Correct slicing
                    sensor_readings.extend(p.getJointReactionForces(self.robotId[env_id], sensor_index, physicsClientId=self.physicsClient[env_id])) # Get Reaction Forces as Accel Data

                except Exception as e:
                    print(f"Error getting sensor data for {sensor_name}: {e}")
                    sensor_readings.extend([0.0] * 6)  # Append zeros if there's an error
            else:
                sensor_readings.extend([0.0] * 6)  # Append zeros if sensor index is None
        return sensor_readings

    def _get_observation(self, env_id):
        joint_states = p.getJointStates(self.robotId[env_id], self.joint_ids[env_id], physicsClientId=self.physicsClient[env_id])
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        # Get base orientation (quaternion)
        base_orientation_quat = p.getBasePositionAndOrientation(self.robotId[env_id], physicsClientId=self.physicsClient[env_id])[1]
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        base_orientation_euler = p.getEulerFromQuaternion(base_orientation_quat)

        # Get sensor readings
        sensor_readings = self._get_sensor_readings(env_id)

        observation = np.array(joint_positions + joint_velocities + list(base_orientation_euler) + sensor_readings,
                               dtype=np.float32)  # Concatenate all observations

        return observation

    def step(self, action):
        observations, rewards, dones, truncateds, infos = [], [], [], [], []
        for env_id in range(self.num_envs):
            scaled_action = np.clip(action[env_id], -1, 1) * self.max_motor_force  # added action scaling
            # Apply action as target motor velocities
            for i, joint_id in enumerate(self.joint_ids[env_id]):
                # *************************************************************************************
                # IMPORTANT: Limit control to leg servo motors (IDs 0 to 9)
                # After inspecting the URDF:
                # Joints 0-9 correspond to the leg servos:
                # 0: csmall1_Revolute-87
                # 1: csmall6_Revolute-88
                # 2: csmall2_Revolute-91
                # 3: servo6_Revolute-94
                # 4: csmall4_Revolute-96
                # 5: servo10_Revolute-98
                # 6: csmall7_Revolute-100
                # 7: servo5_Revolute-103
                # 8: csmall9_Revolute-105
                # 9: servo9_Revolute-107
                if joint_id >= 0 and joint_id <= 9:  # Leg servos
                    p.setJointMotorControl2(self.robotId[env_id], joint_id, controlMode=p.VELOCITY_CONTROL,
                                            targetVelocity=scaled_action[i], force=self.torque, maxVelocity=10.0,
                                            physicsClientId=self.physicsClient[env_id])
                else:
                    # Optionally, set other joints to a default position or velocity. Setting to 0 here.
                    p.setJointMotorControl2(self.robotId[env_id], joint_id, controlMode=p.VELOCITY_CONTROL,
                                            targetVelocity=0, force=0, physicsClientId=self.physicsClient[env_id])

                # *************************************************************************************

            p.stepSimulation(physicsClientId=self.physicsClient[env_id])
            if self.use_gui and env_id == 0: #Only sleep in main gui render
                time.sleep(self.timeStep)  # keep this for rendering mode

            observation = self._get_observation(env_id)
            reward, done = self._get_reward_and_done(env_id)
            info = {}

            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(False)
            infos.append(info)

            # Update episode duration
            if not done:  # Only update if not done
                self.episode_durations[env_id] = time.time() - self.episode_start_time[env_id]
            # print(f"Episode duration: {self.episode_durations[env_id]}")

        return np.array(observations, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), truncateds, infos  # 'truncated' is always False for now

    def estimate_orientation_from_sensors(self, env_id):
        # Get sensor readings
        sensor_readings = self._get_sensor_readings(env_id)
        
        # Initialize arrays for gyro and accelerometer data
        gyro_data = []
        accel_data = []
        
        # Process sensor readings (format: [ang_vel_x, ang_vel_y, ang_vel_z, force_x, force_y, force_z] for each sensor)
        for i in range(len(self.sensor_indices)):
            start_idx = i * 6
            if start_idx + 3 <= len(sensor_readings):
                gyro_data.append(sensor_readings[start_idx:start_idx+3])
            if start_idx + 6 <= len(sensor_readings):
                accel_data.append(sensor_readings[start_idx+3:start_idx+6])
        
        # If no valid sensor data, fall back to PyBullet's orientation
        if not gyro_data or not accel_data:
            base_orientation_quat = p.getBasePositionAndOrientation(self.robotId[env_id], 
                                                                physicsClientId=self.physicsClient[env_id])[1]
            return p.getEulerFromQuaternion(base_orientation_quat)[:2]  # Return roll, pitch
        
        # Average the gyro and accel data across all sensors
        avg_gyro = np.mean(gyro_data, axis=0)
        avg_accel = np.mean(accel_data, axis=0)
        
        # Extract components
        gyro_x, gyro_y, gyro_z = avg_gyro
        accel_x, accel_y, accel_z = avg_accel
        
        # Calculate roll and pitch from accelerometer data
        # Note: This assumes the accelerometer measures gravity when stationary
        accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Avoid division by zero
        if accel_magnitude < 0.001:
            accel_roll = 0
            accel_pitch = 0
        else:
            # Normalize accelerometer readings
            accel_x /= accel_magnitude
            accel_y /= accel_magnitude
            accel_z /= accel_magnitude
            
            # Calculate roll and pitch from accelerometer
            accel_roll = math.atan2(accel_y, accel_z)
            accel_pitch = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2))
        
        # Complementary filter parameters
        # Alpha determines how much we trust the gyro vs accelerometer
        # Higher alpha means more trust in gyro (good for short term)
        # Lower alpha means more trust in accelerometer (good for long term stability)
        alpha = 0.98
        
        # Get the time step for integration
        dt = self.timeStep
        
        # If we have previous estimates, use them; otherwise initialize with accelerometer
        if hasattr(self, 'prev_roll') and hasattr(self, 'prev_pitch'):
            # Integrate gyro data to get change in angle
            gyro_roll = self.prev_roll + gyro_x * dt
            gyro_pitch = self.prev_pitch + gyro_y * dt
            
            # Complementary filter to combine gyro and accelerometer data
            roll = alpha * gyro_roll + (1 - alpha) * accel_roll
            pitch = alpha * gyro_pitch + (1 - alpha) * accel_pitch
        else:
            # First time, just use accelerometer data
            roll = accel_roll
            pitch = accel_pitch
        
        # Store for next iteration
        self.prev_roll = roll
        self.prev_pitch = pitch
        
        return roll, pitch


    def _get_reward_and_done(self, env_id):
        # Get orientation from sensor data (gyroscope and accelerometer)
        roll, pitch = self.estimate_orientation_from_sensors(env_id)
        
        # Get sensor readings for balance assessment
        sensor_readings = self._get_sensor_readings(env_id)
        
        # Extract relevant sensor data for stability assessment
        # We'll use angular velocities from sensors to detect instability
        angular_velocities = []
        reaction_forces = []
        
        # Process sensor readings (format: [ang_vel_x, ang_vel_y, ang_vel_z, force_x, force_y, force_z] for each sensor)
        for i in range(len(self.sensor_indices)):
            start_idx = i * 6
            if start_idx + 3 <= len(sensor_readings):
                angular_velocities.extend(sensor_readings[start_idx:start_idx+3])
            if start_idx + 6 <= len(sensor_readings):
                reaction_forces.extend(sensor_readings[start_idx+3:start_idx+6])
        
        # Calculate stability score based on sensor data
        # Lower angular velocities and balanced forces indicate better stability
        ang_vel_magnitude = np.mean(np.abs(angular_velocities)) if angular_velocities else 0
        force_balance = np.std(reaction_forces) if reaction_forces else 0
        
        # Stability reward: higher when angular velocities are low and forces are balanced
        stability_reward = 1.0 / (1.0 + ang_vel_magnitude + 0.1 * force_balance)
        
        # Upright reward based on sensor-derived orientation (higher when upright)
        upright_reward = math.cos(roll) * math.cos(pitch)
        
        # Time-based reward: increases with standing duration
        # Normalized to 0-1 range and weighted more heavily than in original function
        time_reward = min(self.episode_durations[env_id], MAX_STAND_DURATION) / MAX_STAND_DURATION
        time_reward_weight = 2.0  # Increased weight for time reward
        
        # Check if fallen (pitch or roll past threshold) using sensor-derived orientation
        fallen_threshold = 1.0  # Radians (about 57 degrees)
        done = abs(pitch) > fallen_threshold or abs(roll) > fallen_threshold
        
        # Fall penalty: more negative for quicker falls
        # The quicker the model falls, the higher the negative reward
        if done:
            # Calculate fall speed penalty - higher penalty for falling quickly
            fall_duration = max(0.1, self.episode_durations[env_id])  # Avoid division by zero
            fall_speed_penalty = -5.0 / fall_duration  # More negative for quick falls
            fall_penalty = fall_speed_penalty
        else:
            fall_penalty = 0.0
        
        # Combine all reward components with appropriate weights
        # Note: No velocity penalty as requested by user
        reward = (
            0.5 * upright_reward +         # Weight for orientation
            1.0 * stability_reward +       # Weight for sensor-based stability
            time_reward_weight * time_reward +  # Weight for standing duration
            fall_penalty                   # Fall penalty (only applied when fallen)
        ) * REWARD_SCALE
        
        return reward, done

    def close(self):
        for client in self.physicsClient:
            p.disconnect(physicsClientId=client)


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
        self.episode_rewards = []  # Store rewards per episode
        self.episode_durations = []  # Store durations per episode

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
            'episode_rewards': self.episode_rewards,  # Save episode rewards
            'episode_durations': self.episode_durations,  # Save episode durations
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
        
        # Load episode rewards and durations if available
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_durations = checkpoint.get('episode_durations', [])
        print(f"Model loaded from {filename}")

    def plot_rewards(self, filename):
        """Plots episode rewards over time."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards")
        plt.savefig(filename)
        plt.close()  # Close the plot to prevent display in non-interactive environments

if __name__ == '__main__':
    num_envs = 4  # Use 4 parallel environments
    env = HumanoidStandEnv(render=True, num_envs=num_envs)  # Render set to false for faster training.
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
        noise_clip=NOISE_CLIP,  # Corrected the typo here
        policy_delay=POLICY_DELAY,
        max_action=1.0
    )

    # Load existing model if continuing training
    start_episode = 0  # Keep track of starting episode
    if CONTINUE_TRAINING and os.path.exists(MODEL_SAVE_PATH):
        agent.load(MODEL_SAVE_PATH)
        start_episode = len(agent.episode_rewards)  # Start from the next episode

    # Training loop
    try:
        total_timesteps = 0
        for episode in range(start_episode, EPISODES):  # Start from loaded episode or 0
            states, _ = env.reset()
            episode_rewards = [0] * num_envs  # Initialize rewards for each env
            dones = [False] * num_envs  # Initialize done flags for each env

            for step in range(env.max_episode_steps):
                # Select actions with exploration noise
                actions = [agent.select_action(state) for state in states]
                # Stack actions to have shape (num_envs, action_dim)
                actions = np.stack(actions)
                
                # Perform actions
                next_states, rewards, dones, _, _ = env.step(actions)

                # Store data in replay buffer for each environment
                for i in range(num_envs):
                    agent.replay_buffer.add(states[i], actions[i], next_states[i], rewards[i], float(dones[i]))
                    episode_rewards[i] += rewards[i]  # Accumulate reward for each env

                # Update states
                states = next_states
                total_timesteps += num_envs  # Increment by the number of environments

                # Train agent
                agent.train()
                
                # Visualize the episode in the first environment every VISUALIZE_EPISODE episodes
                if (episode + 1) % VISUALIZE_EPISODE == 0 and step == 0:
                    vis_env = HumanoidStandEnv(render=True, num_envs=1)  # Create new environment for visualization
                    vis_state, _ = vis_env.reset()
                    vis_done = False
                    vis_steps = 0
                    while not vis_done and vis_steps < vis_env.max_episode_steps:
                        vis_action = agent.select_action(vis_state, add_noise=False)  # No exploration noise during visualization
                        vis_next_state, _, vis_done, _, _ = vis_env.step(np.array([vis_action]))  # Step the visualized environment
                        vis_state = vis_next_state
                        vis_steps += 1
                    vis_env.close() # Close Vis Env For the Run
                    print(f"Visualized Episode {episode+1}")


                if all(dones):  # If all environments are done, break the loop
                    break

            # Average reward across all environments for this episode
            avg_reward = np.mean(episode_rewards)
            agent.episode_rewards.append(avg_reward)  # Store the average reward

            # Average duration across all environments for this episode
            avg_duration = np.mean(env.episode_durations)
            agent.episode_durations.append(avg_duration)  # Store the average duration

            print(f"Episode {episode + 1}/{EPISODES} | Avg Reward: {avg_reward:.2f} | Avg Steps: {step + 1} | Avg Duration: {avg_duration:.2f}")

            # Save model and plot rewards periodically
            if (episode + 1) % MODEL_SAVE_INTERVAL == 0:
                agent.save(MODEL_SAVE_PATH)
                agent.plot_rewards(REWARD_PLOT_PATH)  # Plot and save the reward graph

                # Save reward and duration data
                np.save(EPISODE_REWARD_PATH, np.array(agent.episode_rewards))
                np.save(EPISODE_DURATION_PATH, np.array(agent.episode_durations))

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        # Save the model and data at the end of training or interruption
        agent.save(MODEL_SAVE_PATH)
        agent.plot_rewards(REWARD_PLOT_PATH)
        np.save(EPISODE_REWARD_PATH, np.array(agent.episode_rewards))
        np.save(EPISODE_DURATION_PATH, np.array(agent.episode_durations))
        env.close()