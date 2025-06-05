import os
import numpy as np
from scipy.spatial.transform import Rotation

# def spherical_to_camera_pose(radius, phi, theta):
#     """
#     Convert spherical coordinates to camera pose (quaternion and translation).

#     Parameters:
#     - radius: distance from origin
#     - phi: polar angle (in radians, from +Z down)
#     - theta: azimuthal angle (in radians, from +X in XY-plane)

#     Returns:
#     - quat: quaternion [x, y, z, w] representing world-to-camera rotation
#     - t: translation vector, origin of world in camera coordinates
#     """

    
#     theta = np.radians(theta)
#     phi = np.radians(phi)

#     # Convert spherical to Cartesian camera position in world frame
#     cam_pos_world = radius * np.array([
#         np.sin(phi) * np.cos(theta),
#         np.sin(phi) * np.sin(theta),
#         np.cos(phi)
#     ])

#     # Camera is looking toward the origin → Z axis is (origin - position)
#     forward = -cam_pos_world
#     forward /= np.linalg.norm(forward)

#     # Define a default up vector (Z world)
#     up_guess = np.array([0, 0, 1])
#     if np.allclose(forward, up_guess) or np.allclose(forward, -up_guess):
#         up_guess = np.array([0, 1, 0])  # Avoid collinearity

#     # Right = up × forward
#     right = np.cross(up_guess, forward)
#     right /= np.linalg.norm(right)

#     # True up = forward × right
#     up = np.cross(forward, right)

#     # Rotation matrix: columns = camera axes in world frame (R_wc)
#     R_wc = np.column_stack((right, up, forward))

#     # Invert rotation: world → camera
#     R_cw = R_wc.T

#     # Quaternion from world to camera frame
#     rot = R.from_matrix(R_cw)
#     quat = rot.as_quat()  # [x, y, z, w]

#     # Translation: origin of world in camera coordinates = -R_cw @ cam_pos_world
#     t = -R_cw @ cam_pos_world

#     return np.concatenate((quat, t))

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

def create_images_txt(radius: float, input_path: str, output_path: str):
    """
    creates a COLMAP like images.txt file. In this file, the rotations and positions of the cameras are stored as quaternions and a translation vector.
    It follows this format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    
    """
    quats = dir2quats(input_path, radius)
    img_names = os.listdir(input_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(output_path+'/images.txt', 'w') as f:
        f.write('# handmade images.txt file!')

        for i, image_name in enumerate(img_names):
            f.write('\n')
            f.write(str(i)+" ")
            f.write(" ".join(str(x) for x in quats[i]))
            f.write(" 1 ")
            f.write(img_names[i])
            f.write('\n')

def create_cameras_txt(parameters: list[list], output_path: str):
    """
    creates a COLMAP like cameras.txt file.
    It follows this format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    """

    with open(output_path+'/cameras.txt', 'w') as f:
        f.write('# handmade cameras.txt file!')

        for i, camera in enumerate(parameters):
            f.write('\n')
            f.write(str(i+1)+" ")
            f.write(" ".join(str(x) for x in camera))

def create_points3D_txt(output_path: str):
    """
    creates an empty COLMAP like points3D.txt file.
    """
    with open(output_path+'/points3D.txt', 'w') as f:
        pass


if __name__ == "__main__":
    output_path = "datasets/ignore_machine4/known_parameters"
    create_images_txt(200, "datasets/ignore_machine4/images_cropped", output_path)
    parameters = [['SIMPLE_RADIAL', 580, 410, 941.6, 290, 205, -0.6]]
    #parameters = [['SIMPLE_RADIAL', 580, 410, 867.6724, 290, 205, -0.4138]]
    create_cameras_txt(parameters, output_path)
    create_points3D_txt(output_path)