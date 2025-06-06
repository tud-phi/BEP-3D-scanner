from copy import copy
import time
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation
from create_silhouettes import remove_background_white, remove_background_rembg
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import itertools

def project_world_point_on_cam(P_world: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, k: Optional[float]=None) -> tuple[float, float]:
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

    if k is not None:
        u_centered = u - center_pixel[0]
        v_centered = v - center_pixel[1]
        r_squared = u_centered**2 + v_centered**2

        u = u_centered * (1 + k * r_squared)
        v = v_centered * (1 + k * r_squared)
        u = u + center_pixel[0]
        v = v + center_pixel[1]

  
    return v, u


def quaternions2matrices(quaterions: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """converts orientation in quaternion and translation notation to extrinsic camera matrices"""
    qw, qx, qy, qz, tx, ty, tz = quaterions[0], quaterions[1], quaterions[2], quaterions[3], quaterions[4], quaterions[5], quaterions[6]
    rot = Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
    R = rot.as_matrix()
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

def check_points_inside_silhouette(points: np.ndarray, silhouette: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, k: Optional[float] = None) -> tuple[np.ndarray, list]:
    """checks if points in world coordinates fall inside a given binary silhouette"""
    inside_points = np.zeros_like(points)[:,0]
    pixels_shape = (silhouette.shape[0], silhouette.shape[1], 3)
    pixels = np.zeros(pixels_shape)

    for i, point in enumerate(points):
        center_pixel = (int(silhouette.shape[0]), int(silhouette.shape[1]))
        pixel = project_world_point_on_cam(point, K, R, t, k=k)
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


def create_point_grid(radius: float, bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> np.ndarray:
    """Creates a grid of 3D points with equal spacing based on the given radius."""
    spacing = radius

    # Compute number of points along each axis
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]

    x_vals = np.arange(x_min, x_max + spacing, spacing)
    y_vals = np.arange(y_min, y_max + spacing, spacing)
    z_vals = np.arange(z_min, z_max + spacing, spacing)

    # Create a 3D grid
    x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    xyz = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1)

    return xyz

def voxel_carve(points: np.ndarray, silhouettes: np.ndarray, Ks: np.ndarray, quats: np.ndarray, ks: Optional[np.ndarray]=None, score_threshold: float=0.7) -> np.ndarray:
    """carves a 3d reconstruction from a group of candidate points using silhouettes
    
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
    score_threshold: float, default: 0.7
        points that have a score below M * score_threshold are discarded, the rest is kept
        
    Returns
    -------
    np.ndarray
        array of shape (S, 3), containing coordinates of the input points not discarded
    """
    point_scores = np.zeros_like(points)[:,1]
    for i, silhouette in enumerate(silhouettes):
        t1 = time.perf_counter()
        R, t = quaternions2matrices(quats[i])
        if ks is not None:
            k = ks[i]
        else:
            k=None
        inside, _ = check_points_inside_silhouette(points, silhouette, Ks[i], R, t, k=k)
        point_scores += inside
        print(f"processed silhouette {i}: {(time.perf_counter()-t1):.2f}s")

    selected_points = points[point_scores >= int(silhouettes.shape[0] * score_threshold)]

    return selected_points

def remove_inside_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 7 to discard points with all 6 direct neighbors
    # 19 to discard points with all 6 direct neighbors, and all 12 diagonal neighbors
    k = 7
    dists = []
    for i in range(len(pcd.points)):
        _, _, dist = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        dists.append(dist)
        if i == 0:
            lowest = copy(dist)
        else:
            if np.sum(dist) < np.sum(lowest):
                lowest = copy(dist)
    # removes points that do not have the same neighbor distances as the point with the closest neighbors
    lowest_radius = lowest[1]
    lowest_int = [round(x/lowest_radius, 7) for x in lowest]
    dists_int = [[int(x/lowest_radius) for x in dist] for dist in dists]
    points_to_keep = points[[dist != lowest_int for dist in dists_int]]

    #

    return points_to_keep

def expand_point_cloud_with_grid(points: np.ndarray, spacing: float = 1.0, double_offset: bool=False) -> np.ndarray:
    """For each point, creates 26 neighboring points in a 3x3x3 grid (excluding the center),
    avoiding duplicates during generation."""
    
    # Define integer offsets (excluding the center point)

    offset_values = [-2, -1, 0, 1, 2] if double_offset else [-1, 0, 1]

    offsets = [
        (dx, dy, dz)
        for dx in offset_values
        for dy in offset_values
        for dz in offset_values
    ]
    
    seen = set()
    result = []

    for p in points:
        # Convert original point to tuple after snapping to grid
        base = tuple(np.round(p / spacing).astype(int))
        for dx, dy, dz in offsets:
            neighbor = (base[0] + dx, base[1] + dy, base[2] + dz)
            if neighbor not in seen:
                seen.add(neighbor)
                result.append(np.array(neighbor, dtype=float) * spacing)

    return np.array(result)

def iterative_voxel_carving(n_iterations: int, radius: float, points: np.ndarray, silhouettes: np.ndarray, Ks: np.ndarray, quats: np.ndarray, score_threshold: float=0.7) -> np.ndarray:
    for i in range(n_iterations):
        print(f"iteration {i}")

        if i != 0:
            t = time.perf_counter()
            double_offset = True if i == 1 else False
            points = expand_point_cloud_with_grid(points, radius * 0.5**(i), double_offset=double_offset)
            print(f"increase_point_cloud_density: {time.perf_counter()-t}s")

        t = time.perf_counter()
        points = voxel_carve(points, silhouettes, Ks, quats, score_threshold=score_threshold)
        print(f"voxel_carve: {time.perf_counter()-t}s")

        print(f"points in point cloud: {points.shape}")
        save_points_to_ply(points, f'datasets/peer_constant_f/iteration_{i}.ply')

    t = time.perf_counter()
    n_points = points.shape[0]
    points = remove_inside_points(points)
    print(f"removed {-points.shape[0] + n_points} points")
    print(f"remove_inside_points: {time.perf_counter()-t}s")
    
    return points



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

def load_cam_params_from_file(path: str) -> np.ndarray:
    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == '#':
                continue
            f, p_x, p_y = parts[4], parts[5], parts[6]

    return np.array([[f, 0, p_x],
                [0, f, p_y],
                [0,0,1]]).astype(float)


def scale_camera_positions(poses: np.ndarray, measured_distance: float, cam_ids: tuple[int, int]=(0,1)) -> np.ndarray:
    """scales quaternions and translations by comparing the measured distance of two camera positions and the distance of those cameras in 'poses'"""
    R_a, t_a = quaternions2matrices(poses[cam_ids[0]])
    cam_a_coordinates = -R_a.T @ t_a


    R_b, t_b = quaternions2matrices(poses[cam_ids[1]])
    cam_b_coordinates = -R_b.T @ t_b

    difference_vector = cam_a_coordinates - cam_b_coordinates
    distance = np.sqrt(np.dot(difference_vector, difference_vector))
    print(distance)
    poses[:, 4:] *= measured_distance / distance

    return poses


def main():
    t_tot = time.perf_counter()
    dir = "datasets/peer_constant_f/"
    quats, camera_ids, paths = load_poses_from_file(dir+"/known_parameters/images.txt")
    #cam_distance = 20
    #quats = scale_camera_positions(quats, cam_distance)

    # only use images with camera id 2
    #quats = quats[camera_ids==2]
    #paths = list(itertools.compress(paths, camera_ids==2))

    #bounds = ((-7, 7), (-5, 5), (-5, 5))
    bounds = ((-2, 2), (-1.5, 1.5), (-1.5, 1.5))
    radius = (bounds[0][1]-bounds[0][0]) / 15
    grid = create_point_grid(radius, bounds)
    print(grid.shape)

    K = load_cam_params_from_file(dir+"/known_parameters/cameras.txt")
    print(K)
    

    Ks = np.array([K for _ in paths])
    #ks = np.array([k for _ in paths])

    t = time.perf_counter()
    #silhouettes = np.array([remove_background_rembg(dir+"images/"+path) for path in paths])
    silhouettes = np.array([np.array(Image.open(f"datasets/peer_constant_f/silhouettes/sil_{path}")) / 255 for path in paths])
    print(silhouettes.max())
    print(f"silhouettes: {t-time.perf_counter()}s")

    plt.imshow(silhouettes[0], cmap='gray', interpolation='nearest')
    plt.savefig(f"sil_plt.png", dpi=300)

    # t = time.perf_counter()
    # selected_points = voxel_carve(grid, silhouettes, Ks, quats, ks=None, score_threshold=0.7)
    # print(f"voxel_carve: {t-time.perf_counter()}s")
    # n_selected_points = selected_points.shape[0]
    # filtered_points = remove_inside_points(selected_points)

    filtered_points = iterative_voxel_carving(4, radius, grid, silhouettes, Ks, quats, score_threshold=0.7)


    save_points_to_ply(filtered_points, f'{dir}/sfs_reconstruction.ply')
    print(f"shape from silhouettes: {(time.perf_counter() - t_tot):.2f}")


main()

