import os
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import open3d as o3d

def quaternions2matrices(quaterions: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """converts orientation in quaternion and translation notation to extrinsic camera matrices"""
    qw, qx, qy, qz, tx, ty, tz = quaterions[0], quaterions[1], quaterions[2], quaterions[3], quaterions[4], quaterions[5], quaterions[6]
    rot = Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
    R = rot.as_matrix()
    #R = np.linalg.inv(R)
    t = np.array([tx, ty, tz])
    return R, t
 
def quat2euler(quaterions):
    qw, qx, qy, qz, tx, ty, tz = quaterions[0], quaterions[1], quaterions[2], quaterions[3], quaterions[4], quaterions[5], quaterions[6]
    return Rotation.from_quat([qx, qy, qz, qw], scalar_first=False).as_euler("xyz", degrees=True)

def load_poses_from_file(path: str) -> np.ndarray:
    poses = []
    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 9:
                continue  # skip malformed lines
            if parts[0] == '#':
                continue
            # Extract quaternion and translation
            qw = float(parts[1])
            qx = float(parts[2])
            qy = float(parts[3])
            qz = float(parts[4])
            tx = float(parts[5])
            ty = float(parts[6])
            tz = float(parts[7])
            poses.append([qw, qx, qy, qz, tx, ty, tz])
    return np.array(poses)

def plot_poses(pose_array, axis_length=0.6):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Cs = []
    
    for pose in pose_array:
        qw, qx, qy, qz, tx, ty, tz = pose
        
        # Rotation matrix
        rot = Rotation.from_quat([qw, qx, qy, qz], scalar_first=True)
        R_mat = rot.as_matrix()

        # Translation vector
        t = np.array([tx, ty, tz])
        C = -R_mat.T @ t
        Cs.append(C)

        # Plot origin
        ax.scatter(C[0], C[1], C[2], color='black')

        # Draw axes
        x_axis = C + R_mat.T[:, 0] * axis_length
        y_axis = C + R_mat.T[:, 1] * axis_length
        z_axis = C + R_mat.T[:, 2] * axis_length

        ax.plot([C[0], x_axis[0]], [C[1], x_axis[1]], [C[2], x_axis[2]], color='r')  # X: red
        ax.plot([C[0], y_axis[0]], [C[1], y_axis[1]], [C[2], y_axis[2]], color='g')  # Y: green
        ax.plot([C[0], z_axis[0]], [C[1], z_axis[1]], [C[2], z_axis[2]], color='b')  # Z: blue
    #ax.scatter(0, 0, 0, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_xlim([np.min(pose_array[:,4]), np.max(pose_array[:,4])])
    #ax.set_ylim([np.min(pose_array[:,5]), np.max(pose_array[:,5])])
    #ax.set_zlim([np.min(pose_array[:,6]), np.max(pose_array[:,6])])
    elev = 15
    azim = 100
    roll = 0
    ax.view_init(elev=elev, azim=azim, roll=roll)

    ax.set_title('Camera Poses')
    ax.axis('equal')
    plt.savefig(f"poses_{elev}_{azim}_{roll}.png", dpi=300)

    return np.array(Cs)


def spherical_to_camera_pose(radius, phi, theta):
    """
    Convert spherical coordinates to camera pose (quaternion and translation).

    Parameters:
    - radius: distance from origin
    - phi: polar angle (in radians, from +Z down)
    - theta: azimuthal angle (in radians, from +X in XY-plane)

    Returns:
    - quat: quaternion [x, y, z, w] representing world-to-camera rotation
    - t: translation vector, origin of world in camera coordinates
    """

    
    theta = np.radians(theta)
    phi = np.radians(phi)

    # Convert spherical to Cartesian camera position in world frame
    C = radius * np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


    # Forward vector (camera looks toward origin → optical axis = -Z)
    # forward = -C / np.linalg.norm(C)  # from camera to origin
    # down = np.array([0, 0, -1])
    # proj_of_down_on_n = (np.dot(down, forward)) * forward
    # down_orthogonal = down - proj_of_down_on_n
    # down_orthogonal = down_orthogonal / np.linalg.norm(down_orthogonal)
    # right = -np.cross(forward, down_orthogonal)

    # R = np.stack([right, down_orthogonal, forward], axis=1)

    forward = -C / np.linalg.norm(C)  # from camera to origin
    down = np.array([0, 0, -1])
    proj_of_down_on_n = (np.dot(down, forward)) * forward
    down_orthogonal = down - proj_of_down_on_n
    down_orthogonal = down_orthogonal / np.linalg.norm(down_orthogonal)
    right = np.cross(forward, down_orthogonal)

    R = np.stack([down_orthogonal, right, forward], axis=1)
   
    # Convert to quaternion (scipy uses active rotation convention)
    quat = Rotation.from_matrix(R.T).as_quat(scalar_first=True)  # Transpose for world-to-camera (passive)

    # Translation: origin of world in camera coordinates = -R.T * C
    t = -R.T @ C

    return np.concatenate((quat, t))


def dir2quats(path: str, radius: float) -> np.ndarray:
    image_names = os.listdir(path)
    spherical = [image_name.split('.')[0].split('_')[1:] for image_name in image_names]
    return np.array([spherical_to_camera_pose(radius, float(coords[0]), float(coords[1])) for coords in spherical])

def load_poses_from_file(path: str) -> tuple[np.ndarray, np.ndarray, list]:
    """reads images.txt and outputs an array with the camera orientations, a list with camera ids and a list with image paths"""
    poses = []
    cameras = []
    paths = []
    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            if parts[0] == '#':
                continue

            qw = float(parts[1])
            qx = float(parts[2])
            qy = float(parts[3])
            qz = float(parts[4])
            tx = float(parts[5])
            ty = float(parts[6])
            tz = float(parts[7])
            camera = int(parts[8])
            path = parts[9]

            poses.append([qw, qx, qy, qz, tx, ty, tz])
            cameras.append(camera)
            paths.append(path)

    return np.array(poses), np.array(cameras), paths
    

if __name__ == "__main__":
    quats_measured = dir2quats("datasets/ignore_machine4/images_cropped", 2.0)
    quats_colmap = load_poses_from_file("datasets/ignore_machine4/colmap_parameters/images2.txt")[0]
    quats = np.concatenate((quats_measured, quats_colmap))
    print(quats.shape)
    plot_poses(quats, axis_length=0.5)

# # Example input
# poses = np.array([
#     [1, 0, 0, 0, 0, 0, 0],                        # Identity pose
#     [0.707, 0, 0.707, 0, 10, 0, 0],                # 90° around Y, 10 units to the right
#     [0.707, 0.707, 0, 0, 0, 10, 0]                 # 90° around X, 10 units up
# ])

# #quats = load_poses_from_file("datasets/peer_constant_f/known_parameters/images.txt")

# Cs = plot_poses(np.array(quats))
# print(Cs)
# # the plotted poses look incorrect, maybe colmap has another format?

# # update:  the camera coordinates are not tx, ty, tx , but -R.transposed @ t. t is the origin of the world coordinate frame

# # original code to project grid on images still seems correct, but the result is still strange. 

# # the z-axes of the camera frames seem to be pointing towards the world coordinate origin, which is good, but the rotation 
# # of the other axes seem wrong

# # update: rotation is not wrong

# pcd = o3d.io.read_point_cloud("datasets/peer_constant_f/selected_points3.ply")

# points = np.asarray(pcd.points)
# colors = np.zeros_like(pcd.points)

# points = np.vstack([points, Cs])
# new_colors = np.zeros_like(Cs)
# new_colors[:,0] = 1
# old_colors = np.ones_like(colors) 
# colors = np.vstack([old_colors, new_colors])

# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# o3d.io.write_point_cloud('datasets/peer_constant_f/selected_points3_cams.ply', pcd)

# # maybe the coordinate sytem of the projected point is different than the system of the image?
# # when i transpose the silhouette, it works!