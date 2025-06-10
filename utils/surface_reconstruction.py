import open3d as o3d
import numpy as np

def point_cloud2surface(point_cloud, outlier_neighbors=10, outlier_threshold=2, poisson_depth=8):

    # remove isolated outliers
    point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=outlier_neighbors, std_ratio=outlier_threshold)

    # estimate normals for poisson surface reconstruction
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))

    # orient the normals outward
    center = point_cloud.get_center()
    point_cloud.orient_normals_towards_camera_location(center)
    point_cloud.normals = o3d.utility.Vector3dVector(-np.asarray(point_cloud.normals))

    # perform poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=poisson_depth)

    return mesh

def point_cloud_file2surface(input_path, output_path, outlier_neighbors=10, outlier_threshold=2, poisson_depth=8):

    point_cloud = o3d.io.read_point_cloud(input_path)

    # remove isolated outliers
    point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=outlier_neighbors, std_ratio=outlier_threshold)

    # estimate normals for poisson surface reconstruction
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # orient the normals outward
    center = point_cloud.get_center()
    point_cloud.orient_normals_towards_camera_location(center)
    point_cloud.normals = o3d.utility.Vector3dVector(-np.asarray(point_cloud.normals))

    # perform poisson surface reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=poisson_depth)

    o3d.io.write_triangle_mesh(output_path, mesh)

    print('Saved mesh')

if __name__ == "__main__":

    name = 'sfs_reconstructionit4(mooi)'

    pcd = o3d.io.read_point_cloud(f"datasets/peer_constant_f/{name}.ply")

    mesh = point_cloud2surface(pcd)

    o3d.io.write_triangle_mesh(f"datasets/peer_constant_f/{name}_surface.ply", mesh)