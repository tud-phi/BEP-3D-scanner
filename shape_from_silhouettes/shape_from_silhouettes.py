import numpy as np
from scipy.spatial.transform import Rotation
from create_silhouettes import remove_background_white, remove_background_rembg
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import itertools

def project_world_point_on_cam(P_world: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Projects a point in world coordinates on an image
    
    Parameters
    ----------
    P_world: np.ndarray
        array with size (3,) with world coordinates of a point
    K: np.ndarray
        instrinsic camera matrix, size (3,3)
    R: np.ndarray
        rotation matrix from world to camera frame, size (3,3)
    t: np.ndarray
        translation vector, coordinates of the world origin in the camera frame

    Returns
    -------
    tuple[float, float]
        pixel coordinates of P_world
    """
    P_cam = R @ P_world + t
    p = K @ P_cam

    u = p[0] / p[2]
    v = p[1] / p[2]

    return v, u


def quaternions2matrices(quaterions: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """converts orientation in quaternion and translation notation to extrinsic camera matrices"""
    qw, qx, qy, qz, tx, ty, tz = quaterions[0], quaterions[1], quaterions[2], quaterions[3], quaterions[4], quaterions[5], quaterions[6]
    rot = Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
    R = rot.as_matrix()
    print(R)
    t = np.array([tx, ty, tz])
    return R, t

def check_pixel_inside_silhouette(pixel: tuple[float, float], silhouette: np.ndarray) -> bool:
    """checks if a point with pixel coordinates is inside or outside a given binary silhouette"""
    if int(pixel[0]) < 0 or int(pixel[0]) >= silhouette.shape[0]:
        print('outside frame')
        return False
    if int(pixel[1]) < 0 or int(pixel[1]) >= silhouette.shape[1]:
        print('outside frame')
        return False
        
    return silhouette[int(pixel[0]), int(pixel[1])] != 0

def check_points_inside_silhouette(points: np.ndarray, silhouette: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, list]:
    """checks if points in world coordinates fall inside a given binary silhouette"""
    inside_points = np.zeros_like(points)[:,0]
    pixels_shape = (silhouette.shape[0], silhouette.shape[1], 3)
    pixels = np.zeros(pixels_shape)

    for i, point in enumerate(points):
        pixel = project_world_point_on_cam(point, K, R, t)
        if int(pixel[0]) < 0 or int(pixel[0]) >= silhouette.shape[0]:
            continue
        if int(pixel[1]) < 0 or int(pixel[1]) >= silhouette.shape[1]:
            continue
        inside = check_pixel_inside_silhouette(pixel, silhouette)
        inside_points[i] = inside
        if not inside:
            pixels[int(pixel[0]), int(pixel[1])] = np.array([255, 0, 0])
        else:
            pixels[int(pixel[0]), int(pixel[1])] = np.array([255, 255, 255])
    return inside_points, pixels

def create_point_grid(n_points: int, bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> np.ndarray:
    """creates a grid of 3D points"""
    x_vals = np.linspace(bounds[0][0], bounds[0][1], n_points)
    y_vals = np.linspace(bounds[1][0], bounds[1][1], n_points)
    z_vals = np.linspace(bounds[2][0], bounds[2][1], n_points)

    # Create a grid
    x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals)
    xyz = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1)

    return xyz

def voxel_carve(points: np.ndarray, silhouettes: np.ndarray, Ks: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """creates an array 'point_scores' that keeps track for each point in 'points' how many times it fell inside a silhouette
    
    Parameters
    ----------
    points: np.ndarray
        array of shape (N, 3), with candidate points for the reconstructed object
    silhouettes: np.ndarray
        array of shape (M, H, W) with binary masks representing the silhouettes of an object. 0 is outside, 1 inside.
    Ks: np.ndarray
        array of shape (M, 3, 3) with the intrinsic camera matrices corresponding to the silhouettes
    quats: np.ndarray
        array of shape (M, 7) with quaternions and translation information of the camera positions. Format: [QW, QX, QY, QZ, TX, TY, TZ].
        TX, TY, TZ are the coordinates of the world origin in the camera frame. The quaternions represent the rotation from world to camera frame.
    
    Returns
    -------
    np.ndarray
        array of shape (N, s), every point from the input has a score corresponding to the amount of silhouettes it was included in.
    """
    point_scores = np.zeros_like(points)[:,1]
    for i, silhouette in enumerate(silhouettes):
        R, t = quaternions2matrices(quats[i])
        inside, _ = check_points_inside_silhouette(points, silhouette, Ks[i], R, t)
        point_scores += inside
    return point_scores

def save_points_to_ply(points: np.ndarray, filename: str):
    """writes an array with point coordinates to the specified path"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def add_point_to_pointcloud(pcd, new_points, color=None):
    """add points to an open3d point cloud object"""
    points = np.asarray(pcd.points)
    colors = np.zeros_like(pcd.points)
    
    if color is not None:
        points = np.vstack([points, new_points])
        new_colors = np.ones_like(new_points)
    
        new_colors[:] = color
        old_colors = np.ones_like(colors)
        colors = np.vstack([old_colors, new_colors])

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

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


def main():
    quats, camera_ids, paths = load_poses_from_file("datasets/peer_constant_f/known_parameters/images.txt")

    # only use images with camera id 2
    quats = quats[camera_ids==2]
    paths = list(itertools.compress(paths, camera_ids==2))

    bounds = ((-1.5, 1.5), (-0.85, 0.85), (-0.85, 0.85))
    grid = create_point_grid(20, bounds)


    #f = 2700.74
    f = 2708.99
    p_x = 806
    p_y = 604.5

    K = np.array([[f, 0, p_x],
                [0, f, p_y],
                [0,0,1]])
    

    Ks = np.array([K for _ in paths])

    silhouettes = np.array([remove_background_rembg("datasets/peer_constant_f/images/"+path) for path in paths])

    plt.imshow(silhouettes[0], cmap='gray', interpolation='nearest')
    plt.savefig(f"sil_plt.png", dpi=300)

    point_scores = voxel_carve(grid, silhouettes, Ks, quats)
    print(point_scores.max())
    selected_points = grid[point_scores >= int(point_scores.max() * 0.7)]

    # x_vals = bounds[0]
    # y_vals = bounds[1]
    # z_vals = bounds[2]

    # # Cartesian product of all combinations
    # corners = np.array(list(itertools.product(x_vals, y_vals, z_vals)))

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(selected_points)

    # pcd = add_point_to_pointcloud(pcd, corners, np.array([0, 1, 0]))
    
    # o3d.io.write_point_cloud('datasets/peer_constant_f/test.ply', pcd)

    save_points_to_ply(selected_points, 'datasets/peer_constant_f/test.ply')


main()

