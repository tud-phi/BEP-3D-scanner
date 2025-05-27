import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import open3d as o3d

quats = [
        [0.57133172957433154, -0.81660326182624077, 0.057269012182718286, -0.058816900632070367, 0.06691831058771068, -0.067460496226414798, 3.7942352583536048],
        [0.61603289521404159, 0.78137561603527006, -0.033474515216036539, 0.093995082375406644, 0.0038212992141732622, 0.27936470712046985, 3.8791366990108282],
        [0.23070171790546706, 0.96645255518622797, -0.043070241499694635, 0.10436057793360246, -0.034598077411302879, 0.15139390785730958, 3.9874903235340771],
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
        rot = Rotation.from_quat([qx, qy, qz, qw])
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
    ax.scatter(0, 0, 0, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_xlim([np.min(pose_array[:,4]), np.max(pose_array[:,4])])
    #ax.set_ylim([np.min(pose_array[:,5]), np.max(pose_array[:,5])])
    #ax.set_zlim([np.min(pose_array[:,6]), np.max(pose_array[:,6])])
    elev = 45
    azim = 0
    roll = 0
    ax.view_init(elev=elev, azim=azim, roll=roll)

    ax.set_title('Camera Poses')
    ax.axis('equal')
    plt.savefig(f"poses_{elev}_{azim}_{roll}.png", dpi=300)

    return np.array(Cs)


# Example input
poses = np.array([
    [1, 0, 0, 0, 0, 0, 0],                        # Identity pose
    [0.707, 0, 0.707, 0, 10, 0, 0],                # 90° around Y, 10 units to the right
    [0.707, 0.707, 0, 0, 0, 10, 0]                 # 90° around X, 10 units up
])

#quats = load_poses_from_file("datasets/peer_constant_f/known_parameters/images.txt")

Cs = plot_poses(np.array(quats))
print(Cs)
# the plotted poses look incorrect, maybe colmap has another format?

# update:  the camera coordinates are not tx, ty, tx , but -R.transposed @ t. t is the origin of the world coordinate frame

# original code to project grid on images still seems correct, but the result is still strange. 

# the z-axes of the camera frames seem to be pointing towards the world coordinate origin, which is good, but the rotation 
# of the other axes seem wrong

# update: rotation is not wrong

pcd = o3d.io.read_point_cloud("datasets/peer_constant_f/selected_points3.ply")

points = np.asarray(pcd.points)
colors = np.zeros_like(pcd.points)

points = np.vstack([points, Cs])
new_colors = np.zeros_like(Cs)
new_colors[:,0] = 1
old_colors = np.ones_like(colors) 
colors = np.vstack([old_colors, new_colors])

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud('datasets/peer_constant_f/selected_points3_cams.ply', pcd)

# maybe the coordinate sytem of the projected point is different than the system of the image?
# when i transpose the silhouette, it works!