from colmap.reconstruct import reconstruct_unknown_poses, reconstruct_with_known_poses
from colmap.surface_reconstruction import point_cloud2surface
import open3d as o3d



if __name__ == "__main__":

    werkmap = 'ignore_machine5'

    # dit zijn je paths
    database_path = f"datasets/{werkmap}/database.db"
    image_dir = f"datasets/{werkmap}/images/"
    output_path = f"datasets/{werkmap}/"    
    known_parameters_path = f"/workspaces/BEP-3D-scanner/datasets/{werkmap}/known_parameters/"

    #hiermee run je colmap, hij creeert een point cloud (.ply) in je output_path
    reconstruct_unknown_poses(database_path, image_dir, output_path)
    #reconstruct_with_known_poses(database_path, image_dir, output_path, known_parameters_path)

    #hiermee verwijder je outliers, en maak je er een surface van
    name = 'sparse_reconstruction'
    pcd = o3d.io.read_point_cloud(f"datasets/{werkmap}/{name}.ply")
    mesh = point_cloud2surface(pcd)
    o3d.io.write_triangle_mesh(f"datasets/{werkmap}/{name}_surface.ply", mesh)