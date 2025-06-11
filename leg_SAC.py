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
EPISODES = 2000  # Further increased episodes (CRITICAL)
STEPS_PER_EPISODE = 1000
LEARNING_RATE_ACTOR = 7.5e-4  # Slightly Reduced actor learning rate
LEARNING_RATE_CRITIC = 1e-3
DISCOUNT_FACTOR = 0.98
BATCH_SIZE = 128
BUFFER_SIZE = 100000
TAU = 0.007
HIDDEN_LAYER_SIZE = 256
REWARD_SCALE = 5.0  # Reduced from 10.0
ALPHA = 0.2  # Temperature parameter for SAC
AUTO_ENTROPY_TUNING = True  # Enable automatic entropy tuning
LEARNING_RATE_ALPHA = 3e-4
MODEL_SAVE_INTERVAL = 100
MODEL_SAVE_PATH = "humanoid_sac_model_vstable.pth"
REWARD_PLOT_PATH = "reward_plot_sac_vstable.png"
EPISODE_REWARD_PATH = "episode_rewards_sac_vstable.npy"
EPISODE_DURATION_PATH = "episode_durations_sac_vstable.npy"
CONTINUE_TRAINING = True
ren = True
MAX_STAND_DURATION = 10.0
MAX_ACTION_CHANGE = 0.05  # Increased from 0.01 for better responsiveness
NUM_SERVOS = 10

# NEW HYPERPARAMETERS
JOINT_DAMPING = 0.5  # FIXED: Reduced from 0.7 to make joints less rigid
ACTION_L2_REG = 1e-5  # Slightly Reduced L2 regularization
ACTION_L1_REG = 1e-5  # ADDING L1 action regularization
UPRIGHT_REWARD_SCALING = 3.0  # Increased importance of staying upright
FOOT_CONTACT_REWARD = 10.0  # Increased to encourage ground contact
TARGET_VELOCITY_SCALE = 0.5  # Reduced for gentler movements

# New Hyperparameters for Z-Axis Height Reward
HEAD_LINK_INDEX = 19  # Index of the "base_link" link in humanoidV3.urdf (CHECK THIS)
TARGET_HEIGHT_RANGE = (0.32, 1.0)  # Target Z height range (meters)
HEIGHT_REWARD = 1  # Reward for being within the target height range
EXCESSIVE_HEIGHT_PENALTY = -1.0  # Penalty for exceeding maximum height
MAX_HEIGHT = 0.4  # Maximum Z height before penalty
FLYING_PENALTY = -10.0  # VERY HEAVY penalty for flying too high (NEW)
MAX_HEAD_HEIGHT = 0.4  # Height threshold for flying penalty (NEW)

# Constants
MAX_MOTOR_FORCE = 50.0  # FIXED: Reduced from 100.0 to allow more flexible joint movement while still providing enough torque
BASE_POSITION_STABILIZATION_SCALE = 0.1

NUM_AGENTS = 4  # Number of agents


class HumanoidStandEnv(gym.Env):
    def __init__(self, render=ren, num_envs=NUM_AGENTS):  # Modified for multi-agent
        super().__init__()
        self.render = render
        self.num_envs = num_envs
        self.physicsClient = []
        self.use_gui = render

        # Connect to physics servers FIRST
        if self.use_gui:
            # raise Exception("GUI mode is not supported for multi-agent training.  Use DIRECT.")
            self.physicsClient = [p.connect(p.DIRECT) for _ in range(num_envs)]  # Force DIRECT in multi-agent GUI
            print("WARNING: GUI mode with multiple agents can be unstable. Using DIRECT connection for all envs.")


        else:
            self.physicsClient = [p.connect(p.DIRECT) for _ in range(num_envs)]

        # ADDED: Loop through clients to set search path and gravity for each
        for client in self.physicsClient:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
            # Set gravity individually for each environment
            p.setGravity(0, 0, -9.81, physicsClientId=client)

            # Experiment with contact solver parameters
            p.setPhysicsEngineParameter(
                numSolverIterations=100,  # Increase solver iterations (more accurate but slower)
                solverResidualThreshold=1e-6,  # Tighter constraint tolerance
                physicsClientId=client
            )
        # ADDED: Set timestep for each client
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
            "sensor1": 27,
            "sensor2": 12,
            "sensor3": 28,
            "sensor4": 13,
            "sensor5": 29,
            "sensor6": 14,
            "sensor7": 32
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
        orientation = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        self.initial_base_positions = []  # Reset initial base positions list
        for env_id in range(self.num_envs):
            client = self.physicsClient[env_id]
            p.resetSimulation(physicsClientId=client)
            # Set gravity in reset too
            p.setGravity(0, 0, -9.81, physicsClientId=client)

            # FIXED: Ensure URDF path is correct
            urdf_path = os.path.join("PID/half_urdf/humanoidV3_half.urdf")
            # FIXED: Lower initial height to prevent floating
            # Add slight variation in initial X and Y positions for each agent
            initial_x = (env_id % 2) * 1.0 - 0.5  # Spread 2 agents along the X-axis
            initial_y = (env_id // 2) * 1.0 - 0.5  # Spread the other 2 agents along the Y-axis

            self.robotId[env_id] = p.loadURDF(urdf_path, [initial_x, initial_y, 0.0], useFixedBase=False,
                                              baseOrientation=orientation,
                                              flags=p.URDF_USE_SELF_COLLISION, physicsClientId=client)
            # Increase overall mass by 2kg on the base link
            base_link = -1  # The base link has an index of -1
            dynamics_info = p.getDynamicsInfo(self.robotId[env_id], base_link, physicsClientId=client)
            current_mass = dynamics_info[0]

            new_mass = current_mass + 2.0
            p.changeDynamics(self.robotId[env_id], base_link, mass=new_mass, physicsClientId=client)

            # Adjust Ground plane contact properties
            self.ground_id = p.loadURDF("plane.urdf", physicsClientId=client)
            p.changeDynamics(self.ground_id, -1, contactStiffness=1e5, contactDamping=1e4,
                             physicsClientId=client)  # Experiment with stiffness and damping values

            # print(f"reset base_pos: {p.getBasePositionAndOrientation(self.robotId[env_id], physicsClientId=client)[0]}")
            # ------------------------------
            finish_line = 1.0  # Finish line at x = 2.0 units.
            sub_goal_interval = 0.1  # Every 0.1 units is a sub-goal.

            # Draw the finish line as a thick red line.
            p.addUserDebugLine(
                lineFromXYZ=[finish_line, -2, 0.0],
                lineToXYZ=[finish_line, 2, 0.0],
                lineColorRGB=[1, 0, 0],  # Red color.
                lineWidth=3,
                lifeTime=0,  # persist until simulation reset.
                physicsClientId=client
            )

            # Draw sub-goal lines as thinner green lines.
            for sub_goal in np.arange(sub_goal_interval, finish_line, sub_goal_interval):
                p.addUserDebugLine(
                    lineFromXYZ=[sub_goal, -2, 0.0],
                    lineToXYZ=[sub_goal, 2, 0.0],
                    lineColorRGB=[0, 1, 0],  # Green color.
                    lineWidth=1,
                    lifeTime=0,
                    physicsClientId=client
                )
            # ------------------------------

            self.joint_ids[env_id] = []
            num_joints = p.getNumJoints(self.robotId[env_id], physicsClientId=client)
            # print(f"Env {env_id}: Number of joints found: {num_joints}")  # Debug print

            # Get foot link indices
            self.foot_link_indices = []
            for foot_name in self.foot_links:
                for j in range(num_joints):
                    joint_info = p.getJointInfo(self.robotId[env_id], j, physicsClientId=client)
                    # Decode joint name to string for comparison
                    if joint_info[1].decode('utf-8') == foot_name:
                        self.foot_link_indices.append(j)
                        break
            # print(f"Env {env_id}: Foot link indices found: {self.foot_link_indices}")  # Debug print for foot links

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
                        force=self.max_motor_force * 0.8,  # Apply reduced force for initialization
                        physicsClientId=client
                    )
                    p.enableJointForceTorqueSensor(self.robotId[env_id], j, enableSensor=True,
                                                   physicsClientId=client)
                    # Apply joint damping (Corrected function call)
                    p.changeDynamics(self.robotId[env_id], j, physicsClientId=client,
                                     jointDamping=JOINT_DAMPING)  # Apply Damping
            # Debug Print: Joint IDs found for each environment
            print(f"Env {env_id}: Joint IDs: {self.joint_ids[env_id]}")

            if self.action_space is None:
                self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.joint_ids[env_id]),), dtype=np.float32)

            # In your reset() method, update the observation_space calculation:
            if self.observation_space is None:
                base_obs_size = len(self.joint_ids[env_id]) * 2 + 12  # existing dimensions
                sensor_obs_size = len(self.sensor_indices) * 6  # 6 readings per sensor
                obs_size = base_obs_size + sensor_obs_size
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

            # FIXED: Set initial pose with slightly more variation
            initial_pose = [0.0] * len(self.joint_ids[env_id])
            for i, joint_id in enumerate(self.joint_ids[env_id]):
                initial_pose[i] = 0.005 * (random.random() - 0.5)  # Reduced from 0.01 to 0.005
                p.resetJointState(self.robotId[env_id], joint_id, initial_pose[i], 0, physicsClientId=client)

            # Store initial base position
            self.initial_base_positions.append(
                p.getBasePositionAndOrientation(self.robotId[env_id], physicsClientId=client)[0])

            # Increase foot friction
            for foot_index in self.foot_link_indices:
                p.changeDynamics(self.robotId[env_id], foot_index, lateralFriction=1.0,
                                spinningFriction=1.0, restitution=0.0, physicsClientId=client)

            # FIXED: Run a few simulation steps to let the model settle
            for _ in range(100):  # Increase settling steps from 10 to 100
                # No action is taken here, so the humanoid settles due to physics
                p.stepSimulation(physicsClientId=client)

            # Store initial head height and print it
            self.initial_head_height[env_id] = p.getBasePositionAndOrientation(self.robotId[env_id],
                                                                              physicsClientId=client)[0][2]
            # print(f"Initial head height (Env {env_id}): {self.initial_head_height[env_id]}")

            observation = self._get_observation(env_id)
            observations.append(observation)
            infos.append({})

            self.episode_start_time[env_id] = time.time()
            self.episode_durations[env_id] = 0.0
            self.prev_actions[env_id] = np.zeros(len(self.joint_ids[env_id]))

        return np.array(observations, dtype=np.float32), infos

   
    def step(self, action):
        observations, rewards, dones, truncateds, infos = [], [], [], [], []
        for env_id in range(self.num_envs):
            clipped_action = np.clip(action[env_id], -1, 1)
            action_diff = clipped_action - self.prev_actions[env_id]
            clipped_action = self.prev_actions[env_id] + np.clip(
                action_diff, -MAX_ACTION_CHANGE, MAX_ACTION_CHANGE)

            # Add explicit stepSimulation here BEFORE applying motor controls
            p.stepSimulation(physicsClientId=self.physicsClient[env_id])  # Give engine time to catch up

            # FIXED: Use VELOCITY_CONTROL for more natural joint movement
            target_velocities = np.zeros(len(self.joint_ids[env_id]))  # NEW: Create array for ALL target velocities

            for i, joint_id in enumerate(self.joint_ids[env_id]):
                # Scale action to reasonable joint angle range
                target_position = clipped_action[i] * 0.5  # Limit to ±0.5 radians
                print(f"Env {env_id}, Joint {joint_id}: Type(target_position) before getJointState: {type(target_position)}")

                # Get current position
                joint_state = p.getJointState(self.robotId[env_id], joint_id,
                                                physicsClientId=self.physicsClient[env_id])
                current_position = joint_state[0]
                print(f"Env {env_id}, Joint {joint_id}: Type(current_position): {type(current_position)}")

                # Calculate velocity based on position difference
                position_diff = target_position - current_position
                print(f"Env {env_id}, Joint {joint_id}: Type(position_diff): {type(position_diff)}, shape: {np.shape(position_diff) if hasattr(position_diff, 'shape') else 'scalar'}")
            
                target_velocity = position_diff * 3.0  # Reduced gain for smoother motion
                print(f"Env {env_id}, Joint {joint_id}: Type(target_velocity) before assign: {type(target_velocity)}, shape: {np.shape(target_velocity) if hasattr(target_velocity, 'shape') else 'scalar'}")

                # NEW: Assign the calculated velocity to the array, NOT overwriting the scalar
                target_velocities[i] = target_velocity  # Assign to array
                print(f"Env {env_id}, Joint {joint_id}: Type(target_velocities[i]) after assign: {type(target_velocities[i])}, shape: {np.shape(target_velocities[i]) if hasattr(target_velocities[i], 'shape') else 'scalar'}")

            for i, joint_id in enumerate(self.joint_ids[env_id]):  # Second loop, because you need ALL target_velocities now.
                p.setJointMotorControl2(
                    self.robotId[env_id],
                    joint_id,
                    controlMode=p.VELOCITY_CONTROL,  # FIXED: Use velocity control for smoother movement
                    targetVelocity=float(target_velocities[i]),  # Pass the SCALAR value from the array!
                    force=self.max_motor_force,  # Use reduced force from earlier change
                    physicsClientId=self.physicsClient[env_id]
                )

            p.stepSimulation(physicsClientId=self.physicsClient[env_id])

            # Within your step() method (or wherever you update the visualization), do this for each environment:
            for env_id in range(self.num_envs):
                # Correct call: pass the robot id and physics client id; length will use its default or be passed positionally.
                self.draw_orientation_arrow(self.robotId[env_id], self.physicsClient[env_id], length=0.5)

            for env_id in range(self.num_envs):
                self.apply_forward_bias(env_id, force_magnitude=10.0, duration=0.7)

            if self.use_gui and env_id == 0:
                time.sleep(self.timeStep)

            observation = self._get_observation(env_id)
            reward, done = self._get_reward_and_done(clipped_action, env_id)

            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            # truncateds = []  # pure code before castingtruncateds, 
            infos.append({})#  pure code before castingtruncateds, infos.append({})

            # if not done:#if not dones[env_id]:  # pure code before casting

            #    self.episode_durations[env_id] = time.time() - self.episode_start_time[env_id]

            self.prev_actions[env_id] = clipped_action

        return (np.array(observations, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array([*dones], dtype=bool),  # convert dones back to bool before casting
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

        # Original reward calculation...
        upright_reward = (math.cos(roll) * math.cos(pitch)) * UPRIGHT_REWARD_SCALING
        target_height = 0.32  # Approximate standing height
        height_deviation = abs(base_pos[2] - target_height)
        height_penalty = -height_deviation * 5

        if base_pos[2] > 1.0:
            height_penalty -= 10.0

        vertical_velocity = abs(base_velocity[0][2])
        vertical_velocity_penalty = -vertical_velocity * 10.0

        foot_contact_count = 0
        for foot_index in self.foot_link_indices:
            contacts = p.getContactPoints(
                self.robotId[env_id], self.ground_id, linkIndexA=foot_index,
                physicsClientId=self.physicsClient[env_id])
            if len(contacts) > 0:
                foot_contact_count += 2

        foot_contact_reward = 0
        if len(self.foot_link_indices) > 0:
            foot_contact_reward = (foot_contact_count / len(self.foot_link_indices)) * FOOT_CONTACT_REWARD * 2.0
            if foot_contact_count == 0:
                foot_contact_reward -= 10.0

        # NEW: Foot Position Reward
        foot_position_penalty = 0
        for foot_index in self.foot_link_indices:
            foot_state = p.getLinkState(self.robotId[env_id], foot_index, computeLinkVelocity=1,
                                        physicsClientId=self.physicsClient[env_id])
            foot_z = foot_state[0][2]
            foot_position_penalty += max(0, 0.0 - foot_z)
        foot_position_penalty *= 5.0

        initial_base_position = self.initial_base_positions[env_id]
        base_position_distance = np.linalg.norm(
            np.array(base_pos[:2]) - np.array(initial_base_position[:2]))
        base_position_reward = -base_position_distance * BASE_POSITION_STABILIZATION_SCALE

        action_l2_penalty = ACTION_L2_REG * np.sum(np.square(action))
        action_l1_penalty = ACTION_L1_REG * np.sum(np.abs(action))

        angular_velocity_penalty = -np.sum(np.abs(base_velocity[1])) * 0.3

        time_reward = min(self.episode_durations[env_id], MAX_STAND_DURATION) / MAX_STAND_DURATION

        fallen_threshold = 0.8
        head_height_threshold = self.initial_head_height[env_id] * 0.75
        done = abs(pitch) > fallen_threshold or abs(roll) > fallen_threshold or base_pos[
            2] < head_height_threshold
        fall_penalty = -10.0 if done else 0

         # --- Forward Velocity Reward ---
        forward_velocity = base_velocity[0][0]  # x-component of linear velocity
        velocity_reward = max(forward_velocity, 0.0) * 5.0

        # --- Sub-goal Reward (Incremental Progress) ---
        progress_threshold = 0.2  # e.g., every 0.2 m
        current_x = base_pos[0]
        subgoal_reward = 0.0
        if current_x > progress_threshold:
            num_subgoals = int(current_x / progress_threshold)
            subgoal_reward = num_subgoals * 20.0  

        # --- Finish Line Reward ---
        finish_line = 2.0  # Place the finish line further away to force walking
        x_pos = base_pos[0]
        progress = max(0, min(x_pos, finish_line))
        finish_line_reward = (progress / finish_line) * 10.0  
        if x_pos >= finish_line:
            finish_line_reward += 50.0
            done = True

        # Combine the rewards
        reward = (
            0.2 * upright_reward +  # Reduced weight to lower the incentive for staying put
            0.5 * foot_contact_reward +
            0.02 * base_position_reward +
            0.05 * time_reward +
            0.05 * angular_velocity_penalty +
            height_penalty +
            vertical_velocity_penalty +
            fall_penalty -
            action_l2_penalty -
            action_l1_penalty -
            foot_position_penalty
        ) * REWARD_SCALE

        # Add the movement-related rewards
        reward += subgoal_reward + finish_line_reward + velocity_reward
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
        self.obs_dim = obs_dim  # Store obs_dim

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
        # FIX: Use self.obs_dim to create the next_states tensor
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
    env = HumanoidStandEnv(render=ren, num_envs=NUM_AGENTS)  # Modified for multi-agent
    # Debug: Check joint_ids right after environment initialization
    for env_id in range(env.num_envs):
        print(f"Env {env_id} joint_ids length: {len(env.joint_ids[env_id])}")  # Debug print

    # Initialize agents
    agents = [
        SACAgent(
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
        ) for _ in range(NUM_AGENTS)
    ]

    # Load models if continuing training
    if CONTINUE_TRAINING and os.path.exists(MODEL_SAVE_PATH):
        for agent in agents:
            agent.load(MODEL_SAVE_PATH)  # Assuming a single model file
        print(f"Loaded model from {MODEL_SAVE_PATH}")

    episode_rewards = [[] for _ in range(NUM_AGENTS)]
    episode_durations = [[] for _ in range(NUM_AGENTS)]

    for episode in range(EPISODES):
        states, _ = env.reset()
        episode_start_time = time.time()
        cumulative_rewards = np.zeros(NUM_AGENTS)  # Cumulative reward for each agent

        for step in range(STEPS_PER_EPISODE):
            actions = []
            for i, agent in enumerate(agents):
                actions.append(agent.select_action(states[i]))  # Select action for each agent

            actions = np.array(actions)
            next_states, rewards, dones, truncateds, infos = env.step(actions)

            for i, agent in enumerate(agents):
                agent.replay_buffer.add(states[i], actions[i], next_states[i], rewards[i], dones[i])
                cumulative_rewards[i] += rewards[i]  # Accumulate reward for each agent

            states = next_states

            for agent in agents:
                agent.train()  # Train each agent

            if dones.all():
                break

        episode_duration = time.time() - episode_start_time

        for i in range(NUM_AGENTS):
            episode_rewards[i].append(cumulative_rewards[i])
            episode_durations[i].append(episode_duration)
            print(
                f"Episode {episode + 1}/{EPISODES}, Agent {i + 1} Reward: {cumulative_rewards[i]:.2f}, Duration: {episode_duration:.2f}s")

        # Broadcast best agent's weights every 100 episodes
        if (episode + 1) % 100 == 0:
            best_agent_index = np.argmax(cumulative_rewards)
            print(f"Broadcasting weights from agent {best_agent_index + 1}")
            best_agent = agents[best_agent_index]
            for i, agent in enumerate(agents):
                if i != best_agent_index:
                    agent.actor.load_state_dict(best_agent.actor.state_dict())
                    agent.critic.load_state_dict(best_agent.critic.state_dict())
                    agent.critic_target.load_state_dict(best_agent.critic_target.state_dict())
                    # Also copy optimizer states to ensure proper learning rate adaptation
                    agent.actor_optimizer.load_state_dict(best_agent.actor_optimizer.state_dict())
                    agent.critic_optimizer.load_state_dict(best_agent.critic_optimizer.state_dict())
                    if agent.auto_entropy_tuning:
                        agent._log_alpha.data.copy_(best_agent._log_alpha.data)
                        agent.alpha_optim.load_state_dict(best_agent.alpha_optim.state_dict())

        # Save model, plot rewards, etc. every MODEL_SAVE_INTERVAL
        if (episode + 1) % MODEL_SAVE_INTERVAL == 0:
            for agent_idx, agent in enumerate(agents):
                agent.save(f"agent_{agent_idx}_{MODEL_SAVE_PATH}")  # Save each agent's model
                plot_rewards(episode_rewards[agent_idx], f"agent_{agent_idx}_{REWARD_PLOT_PATH}")  # Plot rewards for each agent
                np.save(f"agent_{agent_idx}_{EPISODE_REWARD_PATH}",
                        np.array(episode_rewards[agent_idx]))  # Save rewards for each agent
                np.save(f"agent_{agent_idx}_{EPISODE_DURATION_PATH}",
                        np.array(episode_durations[agent_idx]))  # Save durations for each agent

            print(f"Models saved at episode {episode + 1}")
    # Save final models
    for agent_idx, agent in enumerate(agents):
        agent.save(f"agent_{agent_idx}_{MODEL_SAVE_PATH}")
        plot_rewards(episode_rewards[agent_idx], f"agent_{agent_idx}_{REWARD_PLOT_PATH}")
        np.save(f"agent_{agent_idx}_{EPISODE_REWARD_PATH}", np.array(episode_rewards[agent_idx]))
        np.save(f"agent_{agent_idx}_{EPISODE_DURATION_PATH}", np.array(episode_durations[agent_idx]))
    print("Training completed!")


if __name__ == "__main__":
    main()