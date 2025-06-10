from colmap.reconstruct import reconstruct_unknown_poses
from colmap.surface_reconstruction import point_cloud2surface
import open3d as o3d



if __name__ == "__main__":

    # dit zijn je paths
    database_path = "datasets/peer_constant_f/database.db"
    image_dir = "datasets/peer_constant_f/images/"
    output_path = "datasets/peer_constant_f/"

    #hiermee run je colmap, hij creeert een point cloud (.ply) in je output_path
    reconstruct_unknown_poses(database_path, image_dir, output_path)

    #hiermee verwijder je outliers, en maak je er een surface van
    name = 'sparse_reconstruction'
    pcd = o3d.io.read_point_cloud(f"datasets/peer_constant_f/{name}.ply")
    mesh = point_cloud2surface(pcd)
    o3d.io.write_triangle_mesh(f"datasets/peer_constant_f/{name}_surface.ply", mesh)