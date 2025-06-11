import numpy as np
import matplotlib.pyplot as plt

def to_deg(rad):
    return np.degrees(rad)

def to_rad(deg):
    return np.radians(deg)

def ik_leg_4link(x, y, L):
    L1, L2, L3, L4 = L

    best_sol = None
    min_error = float("inf")

    for theta4_guess in np.radians(np.linspace(-90, 90, 180)):
        # Step 1: Estimate foot orientation by moving back by L4
        x_wrist = x - L4 * np.cos(theta4_guess)
        y_wrist = y - L4 * np.sin(theta4_guess)

        D = np.hypot(x_wrist, y_wrist)
        max_reach = L1 + L2 + L3
        if D > max_reach:
            continue

        # Cosine law for effective "elbow" formed by L2+L3
        cos_q3 = (D**2 - L1**2 - (L2 + L3)**2) / (2 * L1 * (L2 + L3))
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        q3 = np.arccos(cos_q3)

        phi = np.arctan2(y_wrist, x_wrist)
        cos_alpha = (D**2 + L1**2 - (L2 + L3)**2) / (2 * L1 * D)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)

        q1 = phi  # Hip pitch now dynamic
        q2 = -alpha  # Thigh angle downward

        q123 = q1 + q2 + q3
        q4 = theta4_guess - q123  # Ankle compensation

        # Forward kinematics to check final foot position
        angles = [q1, q2, q3, q4]
        pts = forward_kinematics(np.degrees(angles), L)
        tip = pts[-1]
        error = np.linalg.norm(np.array(tip) - np.array([x, y]))

        if error < min_error:
            min_error = error
            best_sol = angles

    if best_sol is None:
        raise ValueError("Target unreachable with any ankle orientation.")

    return [to_deg(a) for a in best_sol]

def forward_kinematics(angles_deg, L):
    angles_rad = np.radians(angles_deg)
    θ1, θ2, θ3, θ4 = angles_rad
    L1, L2, L3, L4 = L

    points = [(0, 0)]
    angle = θ1
    x, y = 0, 0

    for i, (l, a) in enumerate(zip(L, [θ1, θ2, θ3, θ4])):
        angle += a if i != 0 else 0
        x += l * np.cos(angle)
        y += l * np.sin(angle)
        points.append((x, y))

    return points

def plot_leg(joint_positions, target):
    xs, ys = zip(*joint_positions)
    plt.figure(figsize=(6, 8))
    plt.plot(xs, ys, '-o', linewidth=3)
    plt.plot(*target, 'rx', markersize=10, label="Target")
    plt.grid(True)
    plt.axis('equal')
    plt.title("4-Link Humanoid Leg IK")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# -------------------------------
# Test it:
if __name__ == "__main__":
    link_lengths = [9, 8, 6.5, 4.5]  # L1 to L4
    target = (10, -26)  # (x, y) position of foot

    try:
        angles = ik_leg_4link(*target, link_lengths)
        print("Joint Angles (degrees):")
        for i, angle in enumerate(angles, start=1):
            print(f"θ{i} = {angle:.2f}°")

        points = forward_kinematics(angles, link_lengths)
        plot_leg(points, target)

    except ValueError as e:
        print(str(e))
