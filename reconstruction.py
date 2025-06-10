from shape_from_motion.reconstruct import reconstruct_unknown_poses, reconstruct_with_known_poses
from shape_from_silhouettes.shape_from_silhouettes import reconstruct_sfs
from utils.surface_reconstruction import point_cloud_file2surface
import open3d as o3d



if __name__ == "__main__":

    reconstruct_sfs(
        'datasets/machine5/images_cropped/',
        'datasets/machine5/',
        ['SIMPLE_RADIAL', 580, 410, 867.6724, 290, 205, -0.4138],
        #['SIMPLE_RADIAL', 580, 410, 941.6, 290, 205, -0.6],
        n_iterations=4,
        bounds=((-70, 70), (-150, 150), (-150, 150)),
        precision=20,
        background_removal='blue',
        score_threshold=0.9
    )

    point_cloud_file2surface('datasets/machine5/sfs_reconstruction.ply', 'datasets/machine5/sfs_reconstruction_surface.ply')