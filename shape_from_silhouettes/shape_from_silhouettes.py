import numpy as np
from scipy.spatial.transform import Rotation
from create_silhouettes import remove_background_white
from PIL import Image
import open3d as o3d

def project_point(P_world: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Projects a point in world coordinates on an image"""
    P_cam = R @ P_world + t
    p = K @ P_cam

    u = p[0] / p[2]
    v = p[1] / p[2]

    return u, v

def project_points(P_world: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Projects a point in world coordinates on an image"""
    P_cam = R @ P_world.T + t
    p = K @ P_cam

    u = p[0, :] / p[2, :]
    v = p[1, :] / p[2, :]

    return np.column_stack((u, v))


def quaternions2matrices(quaterions: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """converts orientation in quaternion and translation notation to extrinsic camera matrices"""
    qw, qx, qy, qz, tx, ty, tz = quaterions[0], quaterions[1], quaterions[2], quaterions[3], quaterions[4], quaterions[5], quaterions[6]
    rot = Rotation.from_quat([qx, qy, qz, qw], scalar_first=False)
    R = rot.as_matrix()
    #R = np.linalg.inv(R)
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
        pixel = project_point(point, K, R, t)
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

def check_multiple_silhouettes(points, silhouettes, Ks, quats) -> np.ndarray:
    """creates an array 'point_scores' that keeps track for each point in 'points' how many times it fell inside a silhouette"""
    point_scores = np.zeros_like(points)[:,1]
    for i, silhouette in enumerate(silhouettes):
        R, t = quaternions2matrices(quats[i])
        inside, pixels = check_points_inside_silhouette(points, silhouette, Ks[i], R, t)
        point_scores += inside
    return point_scores

def save_points_to_ply(points: np.ndarray, filename: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


def main1():
    bounds = ((-2, 2), (-2, 2), (-2, 2))

    quats = [
        [0.57133172957433154, -0.81660326182624077, 0.057269012182718286, -0.058816900632070367, 0.06691831058771068, -0.067460496226414798, 3.7942352583536048],
        [0.61603289521404159, 0.78137561603527006, -0.033474515216036539, 0.093995082375406644, 0.0038212992141732622, 0.27936470712046985, 3.8791366990108282],
        #[0.23070171790546706, 0.96645255518622797, -0.043070241499694635, 0.10436057793360246, -0.034598077411302879, 0.15139390785730958, 3.9874903235340771],
        #[-0.062870393213185771, 0.99225368300901506, -0.041041249535174185, 0.09897251158203485, 0.084274160340428922, 0.11960455360498207, 4.0504611091577871],
        #[0.40930292759508469, 0.90537467046948072, -0.036230856901334797, 0.10702870896431053, -0.025099895976071367, 0.17160434474739894, 3.9535847385733418],
        #[0.71970658589476533, 0.68923045906380076, -0.028499703922732392, 0.078559349505453596, 0.048899892774562241, 0.25583972227421953, 3.8349755134870365],
        #[-0.25142018094371332, 0.96315525978171612, -0.039721197279900984, 0.086845061204980367, 0.092672695080363174, -0.039761952317510202, 3.9974308645759082],
        #[-0.3712645943338943, 0.92444315941793342, -0.04519629279886811, 0.074328602278781547, 0.07238464552150356, -0.044640719403162547, 3.9241331820036462],
        #[0.843920773417285, 0.53227588704915985, -0.021647213632806656, 0.063336453979814841, 0.069233163619734195, 0.26651497929625334, 3.7613549419943064],
        #[0.98555766803485834, 0.16755696166417461, -0.0060819850256027957, 0.023743568259860806, 0.14713874614173025, 0.11327709460266526, 3.6629367309101837],
        #[0.99999987596450191, 0.00001716552323680, -0.0004781610746267733, 0.0001383412893918577, 0.12230563185152503, -0.019471164434637537, 3.6460223993437038],
        #[0.97328580679805199, -0.22767600216760062, 0.014409006697813899, -0.025898973887514706, 0.15179127900126782, -0.11395109501186045, 3.6577619172509905],
        #[0.89622394096364955, -0.44039593024811324, 0.026251135853152523, -0.046313606325824069, 0.15817930076807374, -0.17192946536764911, 3.7180242143078899]
    ]

    paths = [
        "datasets/peer_constant_f/images/IMG_20250507_144447.jpg",
        "datasets/peer_constant_f/images/IMG_20250507_144459.jpg",
        #"datasets/peer_constant_f/images/IMG_20250507_144456.jpg",
        #"datasets/peer_constant_f/images/IMG_20250507_144453.jpg",
        #"datasets/peer_constant_f/images/IMG_20250507_144457.jpg",
        #"datasets/peer_constant_f/images/IMG_20250507_144459~2.jpg",
        #"datasets/peer_constant_f/images/IMG_20250507_144451.jpg",
        #"datasets/peer_constant_f/images/IMG_20250507_144449.jpg",
        #"datasets/peer_constant_f/images/IMG_20250507_144501.jpg",
        #"datasets/peer_constant_f/images/IMG_20250507_144503.jpg",
        #"datasets/peer_constant_f/images/IMG_20250507_144504.jpg",
        #"datasets/peer_constant_f/images/IMG_20250507_144505.jpg",
        #"datasets/peer_constant_f/images/IMG_20250507_144507.jpg"
    ]

    grid = create_point_grid(50, bounds)


    f = 2700.74
    p_x = 806
    p_y = 604.5

    K = np.array([[f, 0, p_x],
                [0, f, p_y],
                [0,0,1]])

    Ks = [K for _ in paths]

    silhouettes = [remove_background_white(path) for path in paths]

    point_scores = check_multiple_silhouettes(grid, silhouettes, Ks, quats)
    print(point_scores.max())
    selected_points = grid[point_scores >= 1]

    save_points_to_ply(selected_points, 'datasets/peer_constant_f/selected_points1.ply')


main1()

# def main2():
    

#     bounds = ((-31, 31), (-31, 31), (-31, 31))

#     grid = create_point_grid(50, bounds)

#     paths = [
#         "datasets/ignore_mouse/IMG_20250522_134917.jpg",
#         "datasets/ignore_mouse/IMG_20250522_134923.jpg"
#     ]


#     f = 1380
#     p_x = 3024/2
#     p_y = 4032/2

#     K = np.array([[f, 0, p_x],
#                 [0, f, p_y],
#                 [0,0,1]])

#     Ks = [K for _ in paths]

#     quats = [
#         [1, 0, 0, 0, 0, 0, -30],
#         [0, 0, np.sin(np.pi/4), np.cos(np.pi/4), -30, 0, 0]
#     ]

#     silhouettes = [remove_background_white(path) for path in paths]

#     point_scores = check_multiple_silhouettes(grid, silhouettes, Ks, quats)
#     print(point_scores.max())
#     selected_points = grid[point_scores >= 1]

#     save_points_to_ply(selected_points, 'datasets/ignore_mouse/mouse.ply')




