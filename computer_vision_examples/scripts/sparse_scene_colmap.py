import pycolmap

def create_reconstruction(image_dir, output_path, mvs_path, database_path):
    
    pycolmap.extract_features(database_path, image_dir)
    print("extract_features done")
    pycolmap.match_exhaustive(database_path)
    print("match_features done")
    pycolmap.verify_matches(database_path)
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    print("incremental_mapping done")
    maps[0].write(output_path)
    print("wrote map")
    pycolmap.undistort_images(mvs_path, output_path, image_dir)

    # Note: requires compilation with CUDA to run this part for the Dense reconstruction
    # and meshing 
    # ---------------------------------------------------------------------------------# 
    # pycolmap.patch_match_stereo(mvs_path)  
    # ply_path = mvs_path + "dense.ply"
    # pycolmap.stereo_fusion(ply_path, mvs_path)
    # pycolmap.poisson_meshing(output_path, ply_path)

if __name__ == "__main__":
    image_dir = "../../datasets/lego/train/"
    output_path = "../../datasets/lego/train/"

    mvs_path = output_path + "mvs/"
    print("mvs_path: ", mvs_path)
    database_path = output_path + "database.db"
    print("database_path: ", database_path)
    
    create_reconstruction(image_dir, output_path, mvs_path, database_path)
