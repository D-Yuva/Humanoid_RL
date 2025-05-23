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
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters (Tuned for SAC and standing task - VERY STABLE)
EPISODES = 4000  # Further increased episodes (CRITICAL)
STEPS_PER_EPISODE = 200
LEARNING_RATE_ACTOR = 7.5e-5  # Slightly Reduced actor learning rate
LEARNING_RATE_CRITIC = 1e-3
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 256
BUFFER_SIZE = 100000
TAU = 0.005
HIDDEN_LAYER_SIZE = 512
REWARD_SCALE = 5.0  # Reduced from 10.0
ALPHA = 0.2  # Temperature parameter for SAC
AUTO_ENTROPY_TUNING = True  # Enable automatic entropy tuning
LEARNING_RATE_ALPHA = 3e-4
MODEL_SAVE_INTERVAL = 50
MODEL_SAVE_PATH = "humanoid_sac_model_vstable.pth"
REWARD_PLOT_PATH = "reward_plot_sac_vstable.png"
EPISODE_REWARD_PATH = "episode_rewards_sac_vstable.npy"
EPISODE_DURATION_PATH = "episode_durations_sac_vstable.npy"
CONTINUE_TRAINING = False
MAX_STAND_DURATION = 10.0
MAX_ACTION_CHANGE = 0.05  # Increased from 0.01 for better responsiveness
NUM_SERVOS = 17

# NEW HYPERPARAMETERS
JOINT_DAMPING = 0.3  # FIXED: Reduced from 0.7 to make joints less rigid
ACTION_L2_REG = 1e-5  # Slightly Reduced L2 regularization
ACTION_L1_REG = 1e-5  # ADDING L1 action regularization
UPRIGHT_REWARD_SCALING = 3.0  # Increased importance of staying upright
FOOT_CONTACT_REWARD = 5.0  # Increased to encourage ground contact
TARGET_VELOCITY_SCALE = 0.5  # Reduced for gentler movements

# New Hyperparameters for Z-Axis Height Reward
HEAD_LINK_INDEX = 19  # Index of the "base_link" link in humanoidV3.urdf (CHECK THIS)
TARGET_HEIGHT_RANGE = (0.6, 1.0)  # Target Z height range (meters)
HEIGHT_REWARD = 0.1  # Reward for being within the target height range
EXCESSIVE_HEIGHT_PENALTY = -1.0  # Penalty for exceeding maximum height
MAX_HEIGHT = 1.2  # Maximum Z height before penalty
FLYING_PENALTY = -50.0  # VERY HEAVY penalty for flying too high (NEW)
MAX_HEAD_HEIGHT = 1.5  # Height threshold for flying penalty (NEW)

# Constants
MAX_MOTOR_FORCE = 50.0  # FIXED: Reduced from 100.0 to allow more flexible joint movement while still providing enough torque
BASE_POSITION_STABILIZATION_SCALE = 0.1


class HumanoidStandEnv(gym.Env):
    def __init__(self, render=True, num_envs=1):
        super().__init__()
        self.render = render
        self.num_envs = num_envs
        self.physicsClient = []
        self.use_gui = render
        if self.use_gui:
            self.physicsClient.append(p.connect(p.GUI))
            for _ in range(num_envs - 1):
                self.physicsClient.append(p.connect(p.DIRECT))
        else:
            self.physicsClient = [p.connect(p.DIRECT) for _ in range(num_envs)]

        for client in self.physicsClient:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
            # Set gravity individually for each environment
            p.setGravity(0, 0, -9.81, physicsClientId=client)

            # Experiment with contact solver parameters
            p.setPhysicsEngineParameter(
                numSolverIterations=150,  # Increase solver iterations (more accurate but slower)
                solverResidualThreshold=1e-7,  # Tighter constraint tolerance
                physicsClientId=client
            )

        self.timeStep = 1.0 / 240
        for client in self.physicsClient:
            p.setTimeStep(self.timeStep, physicsClientId=client)

        self.max_episode_steps = STEPS_PER_EPISODE
        self.robotId = [None] * num_envs
        self.joint_ids = [[] for _ in range(num_envs)]
        self.num_joints = 0
        self.torque = MAX_MOTOR_FORCE  # Use MAX_MOTOR_FORCE for clarity
        self.max_motor_force = MAX_MOTOR_FORCE
        self.target_velocity_scale = TARGET_VELOCITY_SCALE  # New attribute
        self.action_space = None
        self.observation_space = None
        self.sensor_indices = {
            "sensor1": None,
            "sensor2": None,
            "sensor3": None,
            "sensor4": None,
            "sensor5": None,
            "sensor6": None,
            "sensor7": None
        }
        self.episode_start_time = [0.0] * num_envs
        self.episode_durations = [0.0] * self.num_envs
        self.prev_actions = [np.zeros(NUM_SERVOS) for _ in range(self.num_envs)]
        self.initial_base_positions = []  # Store initial base positions
        self.foot_links = ["ankel1", "ankel2"]  # Names of the foot links
        self.foot_link_indices = []  # Store the link indices of the feet
        self.ground_id = None  # Store the ID of the ground plane
        self.initial_head_height = [0.0] * self.num_envs  # Initialize initial head height

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations = []
        infos = []
        self.initial_base_positions = []  # Reset initial base positions list
        for env_id in range(self.num_envs):
            client = self.physicsClient[env_id]
            p.resetSimulation(physicsClientId=client)
            # Set gravity in reset too
            p.setGravity(0, 0, -9.81, physicsClientId=client)

            # FIXED: Ensure URDF path is correct
            urdf_path = os.path.join("urdf/humanoidV3.urdf")
            # FIXED: Lower initial height to prevent floating
            self.robotId[env_id] = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=False,
                                              flags=p.URDF_USE_SELF_COLLISION, physicsClientId=client)

            # Adjust Ground plane contact properties
            self.ground_id = p.loadURDF("plane.urdf", physicsClientId=client)
            p.changeDynamics(self.ground_id, -1, contactStiffness=1e5, contactDamping=1e4,
                             physicsClientId=client)  # Experiment with stiffness and damping values

            print(f"reset base_pos: {p.getBasePositionAndOrientation(self.robotId[env_id], physicsClientId=client)[0]}")

            self.joint_ids[env_id] = []
            num_joints = p.getNumJoints(self.robotId[env_id], physicsClientId=client)
            print(f"Env {env_id}: Number of joints found: {num_joints}")  # Debug print

            # Get foot link indices
            self.foot_link_indices = []
            for foot_name in self.foot_links:
                for j in range(num_joints):
                    joint_info = p.getJointInfo(self.robotId[env_id], j, physicsClientId=client)
                    # Decode joint name to string for comparison
                    if joint_info[1].decode('utf-8') == foot_name:
                        self.foot_link_indices.append(j)
                        break
            print(f"Env {env_id}: Foot link indices found: {self.foot_link_indices}")  # Debug print for foot links

            # FIXED: Properly initialize all revolute joints
            for j in range(num_joints):
                joint_info = p.getJointInfo(self.robotId[env_id], j, physicsClientId=client)
                if joint_info[2] == p.JOINT_REVOLUTE:
                    self.joint_ids[env_id].append(j)
                    # FIXED: Enable motor control with reduced force for less rigidity
                    p.setJointMotorControl2(
                        self.robotId[env_id],
                        j,
                        controlMode=p.VELOCITY_CONTROL,  # Use velocity control initially
                        targetVelocity=0,
                        force=self.max_motor_force * 0.5,  # Apply reduced force for initialization
                        physicsClientId=client
                    )
                    p.enableJointForceTorqueSensor(self.robotId[env_id], j, enableSensor=True,
                                                   physicsClientId=client)
                    # Apply joint damping (Corrected function call)
                    p.changeDynamics(self.robotId[env_id], j, physicsClientId=client,
                                     jointDamping=JOINT_DAMPING)  # Apply Damping
            print(
                f"Env {env_id}: Number of revolute joints (action space size): {len(self.joint_ids[env_id])}")  # Debug print

            if self.action_space is None:
                self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.joint_ids[env_id]),), dtype=np.float32)

            if self.observation_space is None:
                # Updated observation space size: joints*2 + euler(3) + pos(3) + lin_vel(3) + ang_vel(3)
                obs_size = len(self.joint_ids[env_id]) * 2 + 12
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

            # FIXED: Set initial pose with slightly more variation
            initial_pose = [0.0] * len(self.joint_ids[env_id])
            for i, joint_id in enumerate(self.joint_ids[env_id]):
                initial_pose[i] = 0.005 * (random.random() - 0.5)  # Reduced from 0.01 to 0.005
                p.resetJointState(self.robotId[env_id], joint_id, initial_pose[i], 0, physicsClientId=client)

            # Store initial base position
            self.initial_base_positions.append(
                p.getBasePositionAndOrientation(self.robotId[env_id], physicsClientId=client)[0])

            # FIXED: Run a few simulation steps to let the model settle
            for _ in range(100):  # Increase settling steps from 10 to 100
                # No action is taken here, so the humanoid settles due to physics
                p.stepSimulation(physicsClientId=client)

            # Store initial head height and print it
            self.initial_head_height[env_id] = p.getBasePositionAndOrientation(self.robotId[env_id], physicsClientId=client)[0][2]
            print(f"Initial head height (Env {env_id}): {self.initial_head_height[env_id]}")

            observation = self._get_observation(env_id)
            observations.append(observation)
            infos.append({})

            self.episode_start_time[env_id] = time.time()
            self.episode_durations[env_id] = 0.0
            self.prev_actions[env_id] = np.zeros(len(self.joint_ids[env_id]))

        return np.array(observations, dtype=np.float32), infos

    def _get_sensor_readings(self, env_id):
        sensor_readings = []
        for sensor_name, sensor_index in self.sensor_indices.items():
            if sensor_index is not None:
                try:
                    sensor_state = p.getJointState(self.robotId[env_id], sensor_index,
                                                   physicsClientId=self.physicsClient[env_id])
                    sensor_readings.extend(sensor_state[2:5])
                    sensor_readings.extend(p.getJointReactionForces(self.robotId[env_id], sensor_index,
                                                                   physicsClientId=self.physicsClient[env_id]))
                except Exception as e:
                    print(f"Error getting sensor data for {sensor_name}: {e}")
                    sensor_readings.extend([0.0] * 6)
            else:
                sensor_readings.extend([0.0] * 6)
        return sensor_readings

    def _get_observation(self, env_id):
        joint_states = p.getJointStates(
            self.robotId[env_id], self.joint_ids[env_id],
            physicsClientId=self.physicsClient[env_id])
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        # Use ground truth orientation directly
        base_pos, base_quat = p.getBasePositionAndOrientation(
            self.robotId[env_id], physicsClientId=self.physicsClient[env_id])
        base_euler = p.getEulerFromQuaternion(base_quat)
        base_velocity = p.getBaseVelocity(
            self.robotId[env_id], physicsClientId=self.physicsClient[env_id])

        # Simplified observation space
        observation = np.array(
            joint_positions +
            joint_velocities +
            list(base_euler) +  # Roll, pitch, yaw
            list(base_pos) +  # X, Y, Z position
            list(base_velocity[0]) +  # Linear velocity
            list(base_velocity[1]),  # Angular velocity
            dtype=np.float32
        )
        return observation

    def step(self, action):
        observations, rewards, dones, truncateds, infos = [], [], [], [], []
        for env_id in range(self.num_envs):
            clipped_action = np.clip(action[env_id], -1, 1)
            action_diff = clipped_action - self.prev_actions[env_id]
            clipped_action = self.prev_actions[env_id] + np.clip(
                action_diff, -MAX_ACTION_CHANGE, MAX_ACTION_CHANGE)

            # FIXED: Use VELOCITY_CONTROL for more natural joint movement
            for i, joint_id in enumerate(self.joint_ids[env_id]):
                # Scale action to reasonable joint angle range
                target_position = clipped_action[i] * 0.5  # Limit to ±0.5 radians

                # Get current position
                joint_state = p.getJointState(self.robotId[env_id], joint_id, physicsClientId=self.physicsClient[env_id])
                current_position = joint_state[0]

                # Calculate velocity based on position difference
                position_diff = target_position - current_position
                target_velocity = position_diff * 5.0  # Proportional control

                p.setJointMotorControl2(
                    self.robotId[env_id],
                    joint_id,
                    controlMode=p.VELOCITY_CONTROL,  # FIXED: Use velocity control for smoother movement
                    targetVelocity=target_velocity,  # Set target velocity based on position error
                    force=self.max_motor_force,  # Use reduced force from earlier change
                    physicsClientId=self.physicsClient[env_id]
                )

            p.stepSimulation(physicsClientId=self.physicsClient[env_id])
            if self.use_gui and env_id == 0:
                time.sleep(self.timeStep)

            observation = self._get_observation(env_id)
            reward, done = self._get_reward_and_done(clipped_action, env_id)

            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(False)
            infos.append({})

            if not done:
                self.episode_durations[env_id] = time.time() - self.episode_start_time[env_id]

            self.prev_actions[env_id] = clipped_action

        return (np.array(observations, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=bool),
                truncateds, infos)

    def _get_reward_and_done(self, action, env_id):
        # Use ground truth orientation instead of sensor estimation for stability
        base_orientation_quat = p.getBasePositionAndOrientation(
            self.robotId[env_id], physicsClientId=self.physicsClient[env_id])[1]
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orientation_quat)

        # Get base position and velocity
        base_pos, _ = p.getBasePositionAndOrientation(
            self.robotId[env_id], physicsClientId=self.physicsClient[env_id])
        base_velocity = p.getBaseVelocity(
            self.robotId[env_id], physicsClientId=self.physicsClient[env_id])

        # Core rewards
        upright_reward = (math.cos(roll) * math.cos(pitch)) * UPRIGHT_REWARD_SCALING

        # FIXED: Stronger height penalty to prevent floating
        target_height = 0.85  # Approximate standing height
        height_deviation = abs(base_pos[2] - target_height)
        height_penalty = -height_deviation * 5.0  # Increased from 2.0 to 5.0

        # FIXED: Even stronger penalty for being too high (floating)
        if base_pos[2] > 1.0:  # Reduced from 1.2 to 1.0
            height_penalty -= 50.0  # Increased from 20.0 to 50.0

        # FIXED: Stronger penalty for vertical velocity (prevents floating)
        vertical_velocity = abs(base_velocity[0][2])
        vertical_velocity_penalty = -vertical_velocity * 10.0  # Increased from 5.0 to 10.0

        # FIXED: Enhanced foot contact reward
        foot_contact_count = 0
        for foot_index in self.foot_link_indices:
            contacts = p.getContactPoints(
                self.robotId[env_id], self.ground_id, linkIndexA=foot_index,
                physicsClientId=self.physicsClient[env_id])
            if len(contacts) > 0:
                foot_contact_count += 1

        foot_contact_reward = 0
        if len(self.foot_link_indices) > 0:
            foot_contact_reward = (foot_contact_count / len(self.foot_link_indices)) * FOOT_CONTACT_REWARD * 2.0  # Doubled
            # FIXED: Add penalty for no foot contact
            if foot_contact_count == 0:
                foot_contact_reward -= 10.0  # Strong penalty for no foot contact

        # Base position stability
        initial_base_position = self.initial_base_positions[env_id]
        base_position_distance = np.linalg.norm(
            np.array(base_pos[:2]) - np.array(initial_base_position[:2]))  # Only X,Y
        base_position_reward = -base_position_distance * BASE_POSITION_STABILIZATION_SCALE

        # Action penalties
        action_l2_penalty = ACTION_L2_REG * np.sum(np.square(action))
        action_l1_penalty = ACTION_L1_REG * np.sum(np.abs(action))

        # Angular velocity penalty (reduce spinning)
        angular_velocity_penalty = -np.sum(np.abs(base_velocity[1])) * 0.5

        # Time reward
        time_reward = min(self.episode_durations[env_id], MAX_STAND_DURATION) / MAX_STAND_DURATION

        # Termination conditions
        fallen_threshold = 0.8  # Reduced threshold for more strict standing

        # NEW: Use initial_head_height and adjusted threshold
        head_height_threshold = self.initial_head_height[env_id] * 0.75  # Example: 75% of initial height
        done = abs(pitch) > fallen_threshold or abs(roll) > fallen_threshold or base_pos[2] < head_height_threshold

        fall_penalty = -50.0 if done else 0

        # FIXED: Balanced reward composition with stronger emphasis on height control and foot contact
        reward = (
            0.4 * upright_reward +  # Main reward for staying upright
            0.3 * foot_contact_reward +  # FIXED: Increased from 0.2 to 0.3
            0.15 * base_position_reward +  # Stay in place
            0.1 * time_reward +  # Reward duration
            0.05 * angular_velocity_penalty +  # Reduce spinning
            height_penalty +  # FIXED: Stronger height penalty
            vertical_velocity_penalty +  # FIXED: Stronger vertical velocity penalty
            fall_penalty -  # Penalty for falling
            action_l2_penalty -  # Action regularization
            action_l1_penalty
        ) * REWARD_SCALE

        return reward, done


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

        states = torch.FloatTensor(self.state[ind]).to(self.device)
        actions = torch.FloatTensor(self.action[ind]).to(self.device)
        next_states = torch.FloatTensor(self.next_state[ind]).to(self.device)
        rewards = torch.FloatTensor(self.reward[ind]).to(self.device)
        dones = torch.FloatTensor(self.done[ind]).to(self.device)
        return states, actions, next_states, rewards, dones


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * act_dim)  # Mean and log_std
        )
        self.act_dim = act_dim  # Store act_dim for correct slicing

    def forward(self, state):
        x = self.network(state)
        # Use self.act_dim for splitting mu and log_std
        mu, log_std = x[:, :self.act_dim], x[:, self.act_dim:]
        log_std = torch.clamp(log_std, min=-20, max=2)  # Clamp log_std for stability
        return mu, log_std


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(CriticNetwork, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)


class SACAgent:
    def __init__(self, observation_space, action_space, hidden_size, lr_actor, lr_critic, discount_factor, tau, alpha,
                 auto_entropy_tuning, target_entropy, lr_alpha):
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"SACAgent initialized with obs_dim: {self.obs_dim}, act_dim: {self.act_dim}")  # Debug print

        # Actor-Critic networks
        self.actor = ActorNetwork(self.obs_dim, self.act_dim, hidden_size).to(self.device)
        self.critic = CriticNetwork(self.obs_dim, self.act_dim, hidden_size).to(self.device)
        self.critic_target = CriticNetwork(self.obs_dim, self.act_dim, hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -action_space.shape[0]
            self._log_alpha = torch.zeros(1, requires_grad=True,
                                          device=self.device)
            self.alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)
        else:
            self.alpha = alpha

        self.discount_factor = discount_factor
        self.tau = tau

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, self.obs_dim, self.act_dim)
        self.training_steps = 0
        self.episode_rewards = []
        self.episode_durations = []

    @property
    def alpha_tensor(self):
        if self.auto_entropy_tuning:
            return torch.exp(self._log_alpha).detach()
        else:
            return torch.tensor(self.alpha).to(self.device)

    @property
    def log_alpha(self):
        if self.auto_entropy_tuning:
            return self._log_alpha
        else:
            return torch.log(torch.tensor(self.alpha)).to(self.device)

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            # Ensure state is a tensor and on the correct device.
            # If state is already batched (e.g., from env.reset() or env.step()),
            # its shape will be (num_envs, obs_dim).
            # If it's a single observation, its shape will be (obs_dim,).
            # The ActorNetwork expects (batch_size, obs_dim).
            if state.ndim == 1:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state_tensor = torch.FloatTensor(state).to(self.device)

            mu, log_std = self.actor(state_tensor)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            if evaluate:
                action = torch.tanh(mu)  # Deterministic action for evaluation
            else:
                action = dist.rsample()
            action = torch.tanh(action)
            # Return action with original batch dimension if present.
            # Removed .flatten() as env.step expects (num_envs, action_dim)
            return action.cpu().numpy()

    def train(self):
        if self.replay_buffer.size < BATCH_SIZE:
            return

        self.training_steps += 1
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(BATCH_SIZE)

        # Critic Loss
        q1, q2 = self.critic(states, actions)

        with torch.no_grad():
            next_mu, next_log_std = self.actor(next_states)
            next_std = torch.exp(next_log_std)
            dist = torch.distributions.Normal(next_mu, next_std)
            next_actions = dist.rsample()
            log_prob = dist.log_prob(next_actions).sum(dim=-1, keepdim=True)
            next_actions = torch.tanh(next_actions)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha_tensor * log_prob
            target_q = rewards + (1 - dones) * self.discount_factor * target_q

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Loss
        mu, log_std = self.actor(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        actions_new = dist.rsample()
        log_prob = dist.log_prob(actions_new).sum(dim=-1, keepdim=True)
        actions_new = torch.tanh(actions_new)
        q1_new, q2_new = self.critic(states, actions_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha_tensor * log_prob - q_new).mean()  # Changed the way to calculate actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Entropy Tuning
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        # Soft target update
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self._log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optim.state_dict() if self.auto_entropy_tuning else None,
            'episode_rewards': self.episode_rewards,
            'episode_durations': self.episode_durations,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self._log_alpha = checkpoint['log_alpha']
            self.alpha_optim.load_state_dict(checkpoint['alpha_optimizer'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_durations = checkpoint['episode_durations']


def plot_rewards(rewards, path):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(path)
    plt.close()


def main():
    env = HumanoidStandEnv(render=True, num_envs=1)
    agent = SACAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_size=HIDDEN_LAYER_SIZE,
        lr_actor=LEARNING_RATE_ACTOR,
        lr_critic=LEARNING_RATE_CRITIC,
        discount_factor=DISCOUNT_FACTOR,
        tau=TAU,
        alpha=ALPHA,
        auto_entropy_tuning=AUTO_ENTROPY_TUNING,
        target_entropy=-env.action_space.shape[0],
        lr_alpha=LEARNING_RATE_ALPHA
    )

    if CONTINUE_TRAINING and os.path.exists(MODEL_SAVE_PATH):
        agent.load(MODEL_SAVE_PATH)
        print(f"Loaded model from {MODEL_SAVE_PATH}")

    episode_rewards = []
    episode_durations = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        episode_start_time = time.time()

        for step in range(STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            for i in range(env.num_envs):
                agent.replay_buffer.add(state[i], action[i], next_state[i], reward[i], done[i])

            episode_reward += reward.mean()
            state = next_state

            agent.train()

            if done.all():
                break

        episode_duration = time.time() - episode_start_time
        episode_rewards.append(episode_reward)
        episode_durations.append(episode_duration)

        print(f"Episode {episode + 1}/{EPISODES}, Reward: {episode_reward:.2f}, Duration: {episode_duration:.2f}s")

        if (episode + 1) % MODEL_SAVE_INTERVAL == 0:
            agent.save(MODEL_SAVE_PATH)
            plot_rewards(episode_rewards, REWARD_PLOT_PATH)
            np.save(EPISODE_REWARD_PATH, np.array(episode_rewards))
            np.save(EPISODE_DURATION_PATH, np.array(episode_durations))
            print(f"Model saved at episode {episode + 1}")

    agent.save(MODEL_SAVE_PATH)
    plot_rewards(episode_rewards, REWARD_PLOT_PATH)
    np.save(EPISODE_REWARD_PATH, np.array(episode_rewards))
    np.save(EPISODE_DURATION_PATH, np.array(episode_durations))
    print("Training completed!")

if __name__ == "__main__":
    main()