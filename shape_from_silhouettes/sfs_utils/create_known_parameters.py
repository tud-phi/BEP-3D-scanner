import os
import numpy as np
from scipy.spatial.transform import Rotation

def spherical_to_camera_pose(radius: float, phi: float, theta: float):
    """
    Convert spherical coordinates to camera pose (quaternion and translation).

    Parameters
    ----------
    radius: float
        distance from origin
    phi: float
        polar angle (in radians, from +Z down)
    theta: float
        azimuthal angle (in radians, from +X in XY-plane)

    Returns
    -------
    quats: np.ndarray
        array with quaternions and translation vector in format [QW, QX, QY, QZ, TX, TY, TZ]
    """

    
    theta = np.radians(theta)
    phi = np.radians(phi)

    # Convert spherical to Cartesian camera position in world frame
    C = radius * np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])

    forward = -C / np.linalg.norm(C)  # from camera to origin
    down = np.array([0, 0, -1])
    proj_of_down_on_n = (np.dot(down, forward)) * forward
    down_orthogonal = down - proj_of_down_on_n
    down_orthogonal = down_orthogonal / np.linalg.norm(down_orthogonal)
    right = np.cross(forward, down_orthogonal)

    R = np.stack([down_orthogonal, right, forward], axis=1)
   
    quat = Rotation.from_matrix(R.T).as_quat(scalar_first=True)  # Transpose for world-to-camera (passive)

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
