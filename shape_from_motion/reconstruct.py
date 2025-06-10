import json
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pycolmap
from shape_from_motion.database import COLMAPDatabase
#from database import COLMAPDatabase # if running this file as main
from time import perf_counter
from PIL import Image
from scipy.spatial.transform import Rotation
from shape_from_motion.sfm_utils.colmap_utils import read_images_binary

def open_image_file(path: str) -> list[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    imgs = []
    skip = 0
    for i, line in enumerate(lines):
        if line[0] == '#':
            skip += 1
            continue
        if (i + skip) % 2 == 0:
            line = line[:-1]
            imgs.append(line.split(" "))
    return imgs

def add_imgs_to_database(database_path: str, image_path: str) -> None:
    """adds image paths with id and pose data from images.txt to the database"""
    imgs = open_image_file(image_path)

    db = COLMAPDatabase.connect(database_path)
    for img in imgs:
        if img[0] == '#':
            continue
        db.add_image(img[-1], img[-2], img[0])
    db.commit()
    db.close()

def open_camera_file(path: str) -> list[str]:
    with open(path, 'r') as f:
       lines = f.readlines()
        
    return [line[:-1].split(" ") for line in lines]

def add_cams_to_database(database_path: str, cams_path: str) -> None:
    """adds camera parameters from cameras.txt to the database"""
    cams = open_camera_file(cams_path)

    with open(Path(__file__).parent / "camera_models.json") as f:
        CAMERA_STR_TO_NUMBER = json.load(f)

    db = COLMAPDatabase.connect(database_path)

    for cam in cams:
        if cam[0] == '#':
            continue
        db.add_camera(CAMERA_STR_TO_NUMBER[cam[1]], cam[2], cam[3], cam[4:])
    db.commit()
    db.close()

def create_database(database_path: str) -> None:
    """creates an empty COLMAP database"""
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    
def transpose_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with Image.open(input_path) as img:
                transposed = img.transpose(Image.TRANSPOSE)  # swaps x and y axes
                transposed.save(output_path)


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

def quats_to_cartesian(pose_array: np.ndarray) -> np.ndarray:
    """takes quaternions and translation vectors and returns their corresponding points in cartesian coordinates"""
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

    return np.array(Cs)

def plot_poses(pose_array, axis_length=0.6, angle=[15, 100, 0]):
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
  
    ax.view_init(elev=angle[0], azim=angle[1], roll=angle[2])

    ax.set_title('Camera Poses')
    ax.axis('equal')
    plt.savefig(f"poses_bin.png", dpi=300)

    return np.array(Cs)

def scale_point_cloud(path: str):
    quats = []
    for part in os.listdir(path):
        imgs_part = read_images_binary(path+part+'/images.bin')
        quats_part = np.array([np.concatenate((value.qvec, value.tvec)) for key, value in imgs_part.items()])
        quats.append(quats_part)
    quats = np.concatenate(quats)

    plot_poses(quats, angle=[15, 190, 0])

def reconstruct_with_known_poses(database_path: str, image_dir: str, output_path: str, known_parameters_path: str) -> None:
    """creates a ply file from images and their camera positions."""

    transpose_images(image_dir, output_path+'images_T')

    t = perf_counter()
    create_database(database_path)
    add_imgs_to_database(database_path, f"{known_parameters_path}images.txt")
    add_cams_to_database(database_path, f"{known_parameters_path}cameras.txt")

    pycolmap.extract_features(database_path, output_path+'images_T')
    pycolmap.match_exhaustive(database_path)
    #pycolmap.verify_matches(database_path, (f"{known_parameters_path}/points3D.txt"))

    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(known_parameters_path)
    reconstruction = pycolmap.triangulate_points(reconstruction, database_path, image_dir, output_path)

    reconstruction.export_PLY(f"{output_path}/sparse_reconstruction.ply")
    print(f"reconstruct_with_known_poses took: {perf_counter()-t}s")

def reconstruct_unknown_poses(database_path: str, image_dir: str, output_path: str) -> None:
    t = perf_counter()
    #sift_options = pycolmap.SiftExtractionOptions(peak_threshold=0.001)
    #sift_options = pycolmap.SiftExtractionOptions(domain_size_pooling=True, estimate_affine_shape=True, )
    sift_options = pycolmap.SiftExtractionOptions()
    pycolmap.extract_features(database_path, image_dir, sift_options=sift_options)
    #sift_matching_options = pycolmap.SiftMatchingOptions(guided_matching=True)
    #pycolmap.match_exhaustive(database_path, sift_options=sift_matching_options)
    pycolmap.match_exhaustive(database_path)
    incremental_pipeline_options = pycolmap.IncrementalPipelineOptions(multiple_models=False)
    reconstruction = pycolmap.incremental_mapping(database_path, image_dir, f"{output_path}output/", options=incremental_pipeline_options)
    reconstruction[0].export_PLY(f"{output_path}/sparse_reconstruction.ply")
    print(f"reconstruct_unknown_poses took: {perf_counter()-t}s")

if __name__ == "__main__":
    werkmap = 'datasets/machine5'

    database_path = f"{werkmap}/database.db"
    image_dir = f"{werkmap}/images_cropped/"
    output_path = f"{werkmap}/"
    known_parameters_path = f"{werkmap}/known_parameters/"

    reconstruct_with_known_poses(database_path, image_dir, output_path, known_parameters_path)
    #reconstruct_unknown_poses(database_path, image_dir, output_path)

    #cam_coords_path = f"{werkmap}/output/"
    #scale_point_cloud(cam_coords_path)


