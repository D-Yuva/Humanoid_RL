import pybullet as p
import time
import pybullet_data
import numpy as np
from pid import PID
import os  # Import the 'os' module

URDF_PATH = "half_urdf/humanoidV3_half.urdf"
TOTAL_MASS = 1.2816901245540924 
GRAVITY = -9.81  # m/s^2
CONTROL_FREQUENCY = 50  # Hz
CONTROL_TIMESTEP = 1 / CONTROL_FREQUENCY
SERVO_JOINT_INDICES = [1, 5, 16, 20] 
#SERVO_JOINT_INDICES = [3, 5, 8, 18, 20, 23] 
#Indices of the sensor links in pybullet - CORRECTED
SENSOR_LINK_INDICES = [0, 1, 2, 3, 4, 5, 6]
#SENSOR_LINK_INDICES = [25, 10, 19, 4, 17, 2, 31]
# PID Gains (Tuned for this specific model)
PID_GAINS = {
    "Kp": 0.1, 
    "Ki": 0,  
    "Kd": 0,
    "output_limits": (-1.0, 1.0),  # Motor torque limits
}

# Initialize PyBullet
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, GRAVITY)
p.setTimeStep(CONTROL_TIMESTEP)  # Important for control loop

# Add a plane (ground)
planeId = p.loadURDF("plane.urdf", [0, 0, 0])
robotId = p.loadURDF(URDF_PATH, [0, 0, 0.2], useFixedBase=False)  # Load here after fixing URDF_PATH

# Initialize PID controllers for each servo motor
num_servos = len(SERVO_JOINT_INDICES)
pids = [
    PID(
        Kp=PID_GAINS["Kp"],
        Ki=PID_GAINS["Ki"],
        Kd=PID_GAINS["Kd"],
        output_limits=PID_GAINS["output_limits"],
    )
    for _ in range(num_servos)
]

# Get the base link index
base_link_index = -1  # -1 means the base

def get_base_orientation():
    """
    Get the orientation of the base link (or the robot itself if no base link).
    Returns:
        quaternion (list of 4 floats): [x, y, z, w]
    """
    return p.getBasePositionAndOrientation(robotId)[1]


def get_sensor_orientations():
    """
    Retrieves the orientations of the sensor links.

    Returns:
        orientations (list of tuples): A list of tuples, where each tuple represents the
            orientation of a sensor link as a quaternion [x, y, z, w].  Returns an empty list if there is an error retrieving a link state.
    """
    orientations = []
    for link_index in SENSOR_LINK_INDICES:
        link_state = p.getLinkState(robotId, link_index, computeLinkVelocity=1)
        if link_state is None:  # Check for None
            print(f"ERROR: Could not get link state for link index: {link_index}.  Check if the index is valid and if the link exists in the URDF.")
            return []  # Return an empty list to signal an error
        orientations.append(link_state[1])  # Quaternion orientation
    return orientations


def quaternion_to_euler(quaternion):
    """
    Converts a quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in radians.
    """
    x, y, z, w = quaternion

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw


def calculate_target_joint_positions(sensor_orientations):
    if not sensor_orientations:  # Handle the case of empty sensor_orientations
        return None


    target_positions = []

    pitch_angles = []
    for orientation in sensor_orientations:
        roll, pitch, yaw = quaternion_to_euler(orientation)
        pitch_angles.append(pitch)

    # Use average pitch angle to adjust joint angles
    avg_pitch = np.mean(pitch_angles)

    # Adjust the constants as needed for your robot. These are rough starting points.
    joint_adjust_factor = 0.5  # Scaling factor for pitch to joint angle adjustment

    for i in range(num_servos):
        # Adjust joints in opposite directions to compensate for pitch
        target_position = pids[i].update(avg_pitch, CONTROL_TIMESTEP) * joint_adjust_factor
        target_positions.append(target_position)

    return target_positions

# Simulation loop
try:
    while True:
        # 1. Get Sensor Data (Orientation of Links with Sensors)
        sensor_orientations = get_sensor_orientations()

        if not sensor_orientations:  # Check if sensor data retrieval failed
            print("ERROR: Failed to retrieve sensor orientations.  Exiting simulation loop.")
            break  # Exit the loop

        # 2. Calculate Target Joint Positions (using a balancing algorithm)
        target_joint_positions = calculate_target_joint_positions(sensor_orientations)

        if target_joint_positions is None: # Check if target joint positions calculation failed
            print("ERROR: Failed to calculate target joint positions. Exiting simulation loop.")
            break

        # 3. Apply Control to Servo Motors (PID control)

        for i, joint_index in enumerate(SERVO_JOINT_INDICES):
            # Apply torque directly (more stable for this example)
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=target_joint_positions[i],
            )  # Apply calculated torque

        # 4. PyBullet Simulation Step
        p.stepSimulation()

        # 5. Optional: Visualize Target Positions (e.g., using debug lines)

        # 6. Control Loop Timing (Important for PID performance)
        time.sleep(CONTROL_TIMESTEP)


except p.error as e:
    print(f"PyBullet error: {e}")
except KeyboardInterrupt:
    print("Simulation interrupted by user.")
finally:
    # Clean up
    p.disconnect()
    print("Simulation terminated.")