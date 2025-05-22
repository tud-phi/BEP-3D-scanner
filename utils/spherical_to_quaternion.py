import numpy as np
from scipy.spatial.transform import Rotation as R

def spherical_to_cartesian(radius, theta_deg, phi_deg):
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return np.array([x, y, z])

def look_at_quaternion(position, target=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    forward = target - position
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)
    rot_matrix = np.stack([right, true_up, forward], axis=1)
    
    # Convert rotation matrix to quaternion (in w, x, y, z order)
    rot = R.from_matrix(rot_matrix)
    q = rot.as_quat()  # returns in x, y, z, w order
    qx, qy, qz, qw = q
    return np.array([qw, qx, qy, qz])

def compute_pose(radius, theta_deg, phi_deg):
    translation = spherical_to_cartesian(radius, theta_deg, phi_deg)
    quaternion = look_at_quaternion(translation)
    return np.concatenate([quaternion, translation])

