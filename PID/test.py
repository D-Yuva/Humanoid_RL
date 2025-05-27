import pybullet as p
import pybullet_data
import time
import numpy as np

# Constants
URDF_PATH = "half_urdf/humanoidV3_half.urdf"
SERVO_JOINT_INDICES = [1, 3, 5, 8, 10, 16, 18, 20, 23, 25]
SENSOR_LINK_INDICES = [2, 4, 10, 17, 19, 25, 31]
TARGET_ORIENTATION = [0, 0, 0, 1]  # Target orientation as a quaternion (x, y, z, w)
GRAVITY = -9.81
PLANE_Z = -0.001  # Slightly below the robot's lowest point to avoid initial contact issues

KP = 0.1
KI = 0.001
KD = 0.01

# Scaling factor for control efforts to joint angles
CONTROL_EFFORT_TO_JOINT_ANGLE_SCALE = 0.1

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = np.array([0.0, 0.0, 0.0])  # pitch, roll , yaw
        self.integral = np.array([0.0, 0.0, 0.0])
        self.last_time = time.time()

    def compute(self, current_orientation_euler, target_orientation_euler):
        """
        Computes the control effort based on the current and target orientations.

        Args:
            current_orientation_euler (np.ndarray): Current orientation in Euler angles (roll, pitch, yaw).
            target_orientation_euler (np.ndarray): Target orientation in Euler angles (roll, pitch, yaw).

        Returns:
            np.ndarray: Control efforts for each axis.
        """
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            return np.array([0.0, 0.0, 0.0])  # Avoid division by zero or negative dt

        error = np.array(target_orientation_euler) - np.array(current_orientation_euler)
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        # Control effort calculation
        control_effort = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.previous_error = error
        self.last_time = current_time
        
        return control_effort


class BalancingRobot:
    def __init__(self, urdf_path, servo_joint_indices, sensor_link_indices, target_orientation, gravity, plane_z):
        self.urdf_path = urdf_path
        self.servo_joint_indices = servo_joint_indices
        self.sensor_link_indices = sensor_link_indices
        self.target_orientation = target_orientation
        self.gravity = gravity
        self.plane_z = plane_z
        self.pid_controller = PIDController(KP, KI, KD)
        self.target_orientation_euler = p.getEulerFromQuaternion(target_orientation) # convert quarternion to euler angles
        self.initial_joint_angles = {}  # Store initial joint angles
        self.base_link_index = 32  # Comp3 link index - Assuming this is the base
        self.joint_movements = {joint_index: [] for joint_index in self.servo_joint_indices} #Store movements of the joints

        # Initialize PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.gravity)

        # Load the plane
        p.loadURDF("plane.urdf", [0, 0, self.plane_z])

        # Load the robot
        self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0.1])  
        self.num_joints = p.getNumJoints(self.robot_id)

        # Store initial joint angles
        for i in range(self.num_joints):
            if i in self.servo_joint_indices:
                self.initial_joint_angles[i] = p.getJointState(self.robot_id, i)[0]  # Store initial position


    def reset_joint_angles(self):
        """Resets the robot's joints to a stable configuration (using the stored initial angles)."""
        for joint_index, initial_angle in self.initial_joint_angles.items():
            p.resetJointState(self.robot_id, joint_index, initial_angle)
        # clear joint movements when resetting
        for joint_index in self.servo_joint_indices:
                self.joint_movements[joint_index] = []


    def get_base_orientation_euler(self):
       """Gets the base orientation (Comp3 link) in Euler angles (roll, pitch, yaw)."""
       orientation_quaternion = p.getLinkState(self.robot_id, self.base_link_index, computeForwardKinematics=True)[1]
       orientation_euler = p.getEulerFromQuaternion(orientation_quaternion)
       return orientation_euler
   
    def apply_control_efforts(self, control_efforts):
        """Applies control efforts to the servo joints.

        Args:
            control_efforts (np.ndarray): Control effort for each axis (roll, pitch, yaw).
        """
        max_torque = 1
        # We have 10 servo joints, but the PID controller outputs 3 control efforts (roll, pitch, yaw)
        # We need to map these 3 efforts to the 10 joints.  This is a placeholder;
        # you'll need a proper mapping for your robot's kinematics.
        
        # Example:  Distribute efforts to joints (this is VERY basic and likely needs adaptation)
        for i, joint_index in enumerate(self.servo_joint_indices):
            # This is a placeholder:  How should each joint react to roll, pitch, yaw?
            # This simple example just alternates the sign of the pitch effort
            target_angle = self.initial_joint_angles[joint_index] + (
                control_efforts[0] * CONTROL_EFFORT_TO_JOINT_ANGLE_SCALE +  # Roll influence
                control_efforts[1] * CONTROL_EFFORT_TO_JOINT_ANGLE_SCALE * (-1 if i % 2 else 1) +  # Pitch influence, alternating sign
                control_efforts[2] * CONTROL_EFFORT_TO_JOINT_ANGLE_SCALE   # Yaw influence
            )

            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                controlMode=p.POSITION_CONTROL,  # Use POSITION_CONTROL
                targetPosition=target_angle,  # Set target position
                force=max_torque,  # Limit the torque
            )

            # Store the joint movement
            current_angle = p.getJointState(self.robot_id, joint_index)[0]
            self.joint_movements[joint_index].append(current_angle - self.initial_joint_angles[joint_index])


    def run_simulation(self, duration=20):
        """Runs the simulation for a specified duration and analyzes joint movements."""
        try:  # Add a try-except block to gracefully exit
            start_time = time.time()
            while time.time() - start_time < duration:
                # Get current orientation (roll, pitch, yaw) from the base link (Comp3)
                current_orientation_euler = self.get_base_orientation_euler()  # in radians

                # Compute control efforts using the PID controller
                control_efforts = self.pid_controller.compute(current_orientation_euler, self.target_orientation_euler)

                # Apply control efforts to the joints
                self.apply_control_efforts(control_efforts)

                # Step the simulation
                p.stepSimulation()
                time.sleep(1. / 240.)

        except KeyboardInterrupt:  # exit simulation when Ctrl+C is pressed
            print("Simulation interrupted by user.")
        finally:
            p.disconnect()
            self.analyze_joint_movements()


    def analyze_joint_movements(self):
        """Analyzes and prints the movements of each servo joint."""
        print("\n--- Joint Movement Analysis ---")
        for joint_index, movements in self.joint_movements.items():
            if movements:
                max_movement = np.max(np.abs(movements))
                print(f"Joint {joint_index}: Max Movement = {max_movement:.4f} radians")
            else:
                print(f"Joint {joint_index}: No movement")


def main():
    balancing_robot = BalancingRobot(URDF_PATH, SERVO_JOINT_INDICES, SENSOR_LINK_INDICES, TARGET_ORIENTATION, GRAVITY, PLANE_Z)
    balancing_robot.reset_joint_angles()
    balancing_robot.run_simulation()

if __name__ == "__main__":
    main()