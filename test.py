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
NUM_ROBOTS = 4  # Define the number of robots

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


class HumanoidStandEnv(gym.Env):
    def __init__(self, render=ren, num_robots=NUM_ROBOTS):
        super().__init__()
        self.render = render
        self.num_robots = num_robots  # Number of robots in the environment
        self.use_gui = render
        if self.use_gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient)

        p.setPhysicsEngineParameter(
            numSolverIterations=100,  # Increase solver iterations (more accurate but slower)
            solverResidualThreshold=1e-6,  # Tighter constraint tolerance
            physicsClientId=self.physicsClient
        )

        self.timeStep = 1.0 / 240
        p.setTimeStep(self.timeStep, physicsClientId=self.physicsClient)

        self.max_episode_steps = STEPS_PER_EPISODE
        self.robotId = [None] * num_robots
        self.joint_ids = [[] for _ in range(num_robots)]
        self.num_joints = 0
        self.torque = MAX_MOTOR_FORCE
        self.max_motor_force = MAX_MOTOR_FORCE
        self.target_velocity_scale = TARGET_VELOCITY_SCALE
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
        self.episode_start_time = 0.0  # Single start time for all robots
        self.episode_duration = 0.0    # Single duration for all robots
        self.prev_actions = [np.zeros(NUM_SERVOS) for _ in range(num_robots)]  # Store prev actions for each robot
        self.initial_base_positions = []
        self.foot_links = ["ankel1", "ankel2"]
        self.foot_link_indices = []
        self.ground_id = None
        self.initial_head_height = [] # Initial head height for each robot

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient)

        urdf_path = os.path.join("PID/half_urdf/humanoidV3_half.urdf")

        self.robotId = [None] * self.num_robots  # Initialize robot Ids
        self.joint_ids = [[] for _ in range(self.num_robots)]
        self.initial_base_positions = [] # Reset
        self.initial_head_height = []

        # Adjust Ground plane contact properties
        self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        p.changeDynamics(self.ground_id, -1, contactStiffness=1e5, contactDamping=1e4,
                         physicsClientId=self.physicsClient)

        # Load multiple robots with staggered positions
        orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])
        for i in range(self.num_robots):
            x_offset = (i % 2) * 1.0
            y_offset = (i // 2) * 1.0
            self.robotId[i] = p.loadURDF(urdf_path, [x_offset, y_offset, 0.0], useFixedBase=False, baseOrientation=orientation,
                                          flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.physicsClient)

            # Increase overall mass by 2kg on the base link
            base_link = -1  # The base link has an index of -1
            dynamics_info = p.getDynamicsInfo(self.robotId[i], base_link, physicsClientId=self.physicsClient)
            current_mass = dynamics_info[0]

            new_mass = current_mass + 2.0
            p.changeDynamics(self.robotId[i], base_link, mass=new_mass, physicsClientId=self.physicsClient)


            # Draw the finish line as a thick red line.
            finish_line = 1.0  # Finish line at x = 2.0 units.
            sub_goal_interval = 0.1  # Every 0.1 units is a sub-goal.

            # Draw the finish line as a thick red line.
            p.addUserDebugLine(
                lineFromXYZ=[finish_line + x_offset, -2 + y_offset, 0.0], # Offset Finish line position
                lineToXYZ=[finish_line + x_offset, 2 + y_offset, 0.0],  # Red color.
                lineColorRGB=[1, 0, 0],
                lineWidth=3,
                lifeTime=0,  # persist until simulation reset.
                physicsClientId=self.physicsClient
            )

            # Draw sub-goal lines as thinner green lines.
            for sub_goal in np.arange(sub_goal_interval, finish_line, sub_goal_interval):
                p.addUserDebugLine(
                    lineFromXYZ=[sub_goal + x_offset, -2 + y_offset, 0.0],
                    lineToXYZ=[sub_goal + x_offset, 2 + y_offset, 0.0],
                    lineColorRGB=[0, 1, 0],  # Green color.
                    lineWidth=1,
                    lifeTime=0,
                    physicsClientId=self.physicsClient
                )


            self.joint_ids[i] = []  # Initialize joint IDs for this robot
            num_joints = p.getNumJoints(self.robotId[i], physicsClientId=self.physicsClient)

            # Get foot link indices for this robot
            for foot_name in self.foot_links:
                for j in range(num_joints):
                    joint_info = p.getJointInfo(self.robotId[i], j, physicsClientId=self.physicsClient)
                    if joint_info[1].decode('utf-8') == foot_name:
                        self.foot_link_indices.append((i, j))  # Store robot index and joint index
                        break

            # FIXED: Properly initialize all revolute joints for this robot
            for j in range(num_joints):
                joint_info = p.getJointInfo(self.robotId[i], j, physicsClientId=self.physicsClient)
                if joint_info[2] == p.JOINT_REVOLUTE:
                    self.joint_ids[i].append(j)
                    p.setJointMotorControl2(
                        self.robotId[i],
                        j,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0,
                        force=self.max_motor_force * 0.8,
                        physicsClientId=self.physicsClient
                    )
                    p.enableJointForceTorqueSensor(self.robotId[i], j, enableSensor=True,
                                                   physicsClientId=self.physicsClient)
                    p.changeDynamics(self.robotId[i], j, physicsClientId=self.physicsClient,
                                     jointDamping=JOINT_DAMPING)

            # Set initial pose
            initial_pose = [0.0] * len(self.joint_ids[i])
            for j, joint_id in enumerate(self.joint_ids[i]):
                initial_pose[j] = 0.005 * (random.random() - 0.5)
                p.resetJointState(self.robotId[i], joint_id, initial_pose[j], 0, physicsClientId=self.physicsClient)

            # Store initial base position
            self.initial_base_positions.append(p.getBasePositionAndOrientation(self.robotId[i], physicsClientId=self.physicsClient)[0])

            # Increase foot friction
            for robot_index, foot_index in self.foot_link_indices:  #Correctly iterate the list of tuples
                if robot_index == i:
                    p.changeDynamics(self.robotId[i], foot_index, lateralFriction=1.0,
                                spinningFriction=1.0, restitution=0.0, physicsClientId=self.physicsClient)

            # Store initial head height for this robot
            self.initial_head_height.append(p.getBasePositionAndOrientation(self.robotId[i], physicsClientId=self.physicsClient)[0][2])

        # Run a few simulation steps to let the robots settle
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physicsClient)

        # Define action and observation space AFTER loading the robots
        if self.action_space is None:
            self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.joint_ids[0]),), dtype=np.float32)
        if self.observation_space is None:
            base_obs_size = len(self.joint_ids[0]) * 2 + 12  # existing dimensions
            sensor_obs_size = len(self.sensor_indices) * 6        # 6 readings per sensor
            obs_size = base_obs_size + sensor_obs_size
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size * self.num_robots,), dtype=np.float32) #For all Robots

        observation = self._get_observation()
        info = {}  # Single info dictionary for the environment
        self.episode_start_time = time.time() #Single start time
        self.episode_duration = 0.0 #Single Duration
        self.prev_actions = [np.zeros(len(self.joint_ids[i])) for i in range(self.num_robots)]  # Reset previous actions


        return observation, info

    def _get_sensor_readings(self, robot_id):
        sensor_readings = []
        for sensor_name, sensor_index in self.sensor_indices.items():
            if sensor_index is not None:
                try:
                    # Get the joint state, which returns a tuple of:
                    # (position, velocity, reactionForces, appliedTorque)
                    sensor_state = p.getJointState(
                        self.robotId[robot_id], sensor_index, physicsClientId=self.physicsClient
                    )
                    # Extract the reaction forces (assumed to be a tuple or list of floats)
                    reaction_forces = sensor_state[2]
                    # Ensure that reaction_forces is a flat list of floats
                    sensor_readings.extend([float(x) for x in reaction_forces])
                except Exception as e:
                    print(f"Error getting sensor data for {sensor_name}: {e}")
                    # If an error occurs, append zeros in place of 6 readings
                    sensor_readings.extend([0.0] * 6)
            else:
                # If sensor index is None, use zeros
                sensor_readings.extend([0.0] * 6)
        return sensor_readings

    def _get_observation(self):
        observations = []
        for robot_id in range(self.num_robots):  # Get observations for all robots
            joint_states = p.getJointStates(
                self.robotId[robot_id], self.joint_ids[robot_id],
                physicsClientId=self.physicsClient)
            joint_positions = [state[0] for state in joint_states]
            joint_velocities = [state[1] for state in joint_states]

            base_pos, base_quat = p.getBasePositionAndOrientation(
                self.robotId[robot_id], physicsClientId=self.physicsClient)
            base_euler = p.getEulerFromQuaternion(base_quat)
            base_velocity = p.getBaseVelocity(
                self.robotId[robot_id], physicsClientId=self.physicsClient)

            base_observation = np.array(
                joint_positions +
                joint_velocities +
                list(base_euler) +
                list(base_pos) +
                list(base_velocity[0]) +
                list(base_velocity[1]),
                dtype=np.float32
            )

            sensor_readings = np.array(self._get_sensor_readings(robot_id), dtype=np.float32)  # Get sensors for this robot
            observation = np.concatenate([base_observation, sensor_readings])
            observations.extend(observation)  # Append this robot's observation
        return np.array(observations, dtype=np.float32)  # Single flattened array


    def draw_orientation_arrow(self, robot_id, length=0.5):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robotId[robot_id], physicsClientId=self.physicsClient)
        # Using [0, -length, 0] ensures that after the 90° rotation, the arrow points along +x.
        forward_local = [0, -length, 0]
        forward_world, _ = p.multiplyTransforms([0, 0, 0], base_quat, forward_local, [0, 0, 0, 1])
        arrow_end = [base_pos[i] + forward_world[i] for i in range(3)]
        p.addUserDebugLine(
            base_pos,
            arrow_end,
            lineColorRGB=[1, 0, 0],  # Red arrow for visibility
            lineWidth=2,
            lifeTime=0.1,  # Short lifetime to refresh each simulation step
            physicsClientId=self.physicsClient
        )

    def apply_forward_bias(self, robot_id, force_magnitude=10.0, duration=0.5):
        # Only apply for the first 'duration' seconds of the episode
        if self.episode_duration < duration: # Used Single Episode Duration
            base_pos, _ = p.getBasePositionAndOrientation(self.robotId[robot_id], physicsClientId=self.physicsClient)
            # Apply the force in the positive x direction in WORLD_FRAME
            p.applyExternalForce(
                self.robotId[robot_id],
                linkIndex=-1,  # typically the base link has index -1
                forceObj=[force_magnitude, 0, 0],
                posObj=base_pos,
                flags=p.WORLD_FRAME,
                physicsClientId=self.physicsClient
            )


    def step(self, action):
        # Centralized control: apply the same action to all robots
        for i in range(self.num_robots): #apply the same action to all robots
            clipped_action = np.clip(action, -1, 1)  # Clip the single action
            action_diff = clipped_action - self.prev_actions[i]
            clipped_action = self.prev_actions[i] + np.clip(action_diff, -MAX_ACTION_CHANGE, MAX_ACTION_CHANGE)

            for j, joint_id in enumerate(self.joint_ids[i]):
                target_position = clipped_action[j] * 0.5
                joint_state = p.getJointState(self.robotId[i], joint_id, physicsClientId=self.physicsClient)
                current_position = joint_state[0]
                position_diff = target_position - current_position
                target_velocity = position_diff * 3.0

                p.setJointMotorControl2(
                    self.robotId[i],
                    joint_id,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=target_velocity,
                    force=self.max_motor_force,
                    physicsClientId=self.physicsClient
                )
            self.prev_actions[i] = clipped_action

        p.stepSimulation(physicsClientId=self.physicsClient)

        # Within your step() method (or wherever you update the visualization), do this for each environment:
        for robot_id in range(self.num_robots):
            self.draw_orientation_arrow(robot_id)

        for robot_id in range(self.num_robots): # Apply forward bias to all robots
            self.apply_forward_bias(robot_id, force_magnitude=10.0, duration=0.7)


        if self.use_gui:
            time.sleep(self.timeStep)

        next_state = self._get_observation()
        rewards, done = self._get_reward_and_done(action) # Get individual rewards
        self.episode_duration = time.time() - self.episode_start_time # Calculate episode Duration using single episode start time


        truncated = False # no truncations.
        info = {"individual_rewards": rewards}  # Pass individual rewards in info

        return next_state, sum(rewards) / self.num_robots, done, truncated, info  # Total reward for environment, done



    def _get_reward_and_done(self, action):
        robot_rewards = []  # Store individual rewards
        done = False
        for robot_id in range(self.num_robots):
            base_orientation_quat = p.getBasePositionAndOrientation(
                self.robotId[robot_id], physicsClientId=self.physicsClient)[1]
            roll, pitch, yaw = p.getEulerFromQuaternion(base_orientation_quat)

            base_pos, _ = p.getBasePositionAndOrientation(
                self.robotId[robot_id], physicsClientId=self.physicsClient)
            base_velocity = p.getBaseVelocity(
                self.robotId[robot_id], physicsClientId=self.physicsClient)

            upright_reward = (math.cos(roll) * math.cos(pitch)) * UPRIGHT_REWARD_SCALING

            target_height = 0.32
            height_deviation = abs(base_pos[2] - target_height)
            height_penalty = -height_deviation * 5

            if base_pos[2] > 1.0:
                height_penalty -= 10.0

            vertical_velocity = abs(base_velocity[0][2])
            vertical_velocity_penalty = -vertical_velocity * 10.0

            foot_contact_count = 0
            for robot_index, foot_index in self.foot_link_indices: #Foot Link Incides store with the Robot id
                 if robot_index == robot_id:
                    contacts = p.getContactPoints(
                        self.robotId[robot_id], self.ground_id, linkIndexA=foot_index,
                        physicsClientId=self.physicsClient)
                    if len(contacts) > 0:
                        foot_contact_count += 2

            foot_contact_reward = 0
            if len(self.foot_links) > 0: # Use len(self.foot_links)
                foot_contact_reward = (foot_contact_count / len(self.foot_links)) * FOOT_CONTACT_REWARD * 2.0 # Use len(self.foot_links) instead of self.foot_link_indices.
                if foot_contact_count == 0:
                    foot_contact_reward -= 10.0

            # NEW: Foot Position Reward
            foot_position_penalty = 0
            for robot_index, foot_index in self.foot_link_indices: # iterate the tuples.
                if robot_index == robot_id: # Only calculate for current robot.
                    foot_state = p.getLinkState(self.robotId[robot_id], foot_index, computeLinkVelocity=1,
                                                physicsClientId=self.physicsClient)
                    foot_z = foot_state[0][2]
                    foot_position_penalty += max(0, 0.0 - foot_z) #foot pos penalty

            foot_position_penalty *= 5.0


            initial_base_position = self.initial_base_positions[robot_id]
            base_position_distance = np.linalg.norm(
                np.array(base_pos[:2]) - np.array(initial_base_position[:2]))
            base_position_reward = -base_position_distance * BASE_POSITION_STABILIZATION_SCALE

            action_l2_penalty = ACTION_L2_REG * np.sum(np.square(action)) #Single action applied
            action_l1_penalty = ACTION_L1_REG * np.sum(np.abs(action))

            angular_velocity_penalty = -np.sum(np.abs(base_velocity[1])) * 0.3

            time_reward = min(self.episode_duration, MAX_STAND_DURATION) / MAX_STAND_DURATION #Usign single episode duration

            fallen_threshold = 0.8
            head_height_threshold = self.initial_head_height[robot_id] * 0.75 # Robot Specific Head Height
            robot_done = abs(pitch) > fallen_threshold or abs(roll) > fallen_threshold or base_pos[2] < head_height_threshold

            fall_penalty = -10.0 if robot_done else 0

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
                robot_done = True

            reward = (
                0.2 * upright_reward +
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

            reward += subgoal_reward + finish_line_reward + velocity_reward
            robot_rewards.append(reward)  # Append the reward
            

            # done is True if ANY robot is done, else remains False
            done = done or robot_done #Done Should depend on each robots, so if any is done, then end the episode.

        return robot_rewards, done #return the list of rewards

class ReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, NUM_ROBOTS), dtype=np.float32) #Change: Store rewards for each robot
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward # Store the entire rewards array
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        states = torch.FloatTensor(self.state[ind]).to(self.device)
        actions = torch.FloatTensor(self.action[ind]).to(self.device)
        next_states = torch.FloatTensor(self.next_state[ind]).to(self.device)
        rewards = torch.FloatTensor(self.reward[ind]).to(self.device) # Returns individual rewards
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
                 auto_entropy_tuning, target_entropy, lr_alpha, num_robots): #Added num_robots
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_robots = num_robots #Store num_robots
        print(f"SACAgent initialized with obs_dim: {self.obs_dim}, act_dim: {self.act_dim}, num_robots: {self.num_robots}")

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
            if state.ndim == 1:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Use unsqueeze for a single state
            else:
                state_tensor = torch.FloatTensor(state).to(self.device)
            mu, log_std = self.actor(state_tensor)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            if evaluate:
                action = torch.tanh(mu)
            else:
                action = dist.rsample()
            action = torch.tanh(action)
            return action.cpu().numpy().flatten() #Flatten action to one dimension.


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
            # Use individual rewards here instead of summing or averaging them
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
    env = HumanoidStandEnv(render=ren)  # Single environment with multiple robots
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
        lr_alpha=LEARNING_RATE_ALPHA,
        num_robots=NUM_ROBOTS  # Pass num_robots to SACAgent
    )

    if CONTINUE_TRAINING and os.path.exists(MODEL_SAVE_PATH):
        agent.load(MODEL_SAVE_PATH)
        print(f"Loaded model from {MODEL_SAVE_PATH}")

    episode_rewards = []
    episode_durations = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        episode_start_time = time.time() #Single start time for all
        step = 0 # steps reset in the Episode.

        #Loop only ends with "done", all other Robots should be running.
        while True:
            step += 1 # increment the stepps in loop

            action = agent.select_action(state) #Action for the Single centralized agent
            next_state, reward, done, _, info = env.step(action) # Get individual rewards

            agent.replay_buffer.add(state, action, next_state, reward, done) #Store individual rewards in the buffer

            #Corrected line: sum the rewards list before adding to episode_reward
            episode_reward += reward

            state = next_state #set the next action
            agent.train() #Train Agent

            if done or step >= STEPS_PER_EPISODE:
                break


        episode_duration = time.time() - episode_start_time # Get duration of the Single episode with time.
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