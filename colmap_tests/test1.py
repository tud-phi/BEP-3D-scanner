import pycolmap
from database import COLMAPDatabase

def open_image_file(path: str) -> list[str]:
    with open(path, 'r') as f:
       lines = f.readlines()
    imgs = []
    for i, line in enumerate(lines):
       if i % 2 == 0:
           line = line[:-1]
           imgs.append(line.split(" "))
    return imgs

def add_imgs_to_database(database_path: str, image_path: str) -> None:
    """adds image paths with id and pose data from images.txt to the database"""
    imgs = open_image_file(image_path)
    db = COLMAPDatabase.connect(database_path)
    #db.create_tables()
    for img in imgs:
        db.add_image(img[-1], img[-2], img[0])
    db.commit()
    db.close()

def open_camera_file(path: str) -> list[str]:
    with open(path, 'r') as f:
       lines = f.readlines()
        
    return [line.split(" ") for line in lines]

def add_cams_to_database(database_path: str, cams_path: str) -> None:
    """adds camera parameters from cameras.txt to the database"""
    cams = open_camera_file(cams_path)
    db = COLMAPDatabase.connect(database_path)
    for cam in cams:
        db.add_camera(cam[1], cam[2], cam[3], cam[4:7])
    db.commit()
    db.close()

def create_database(database_path: str) -> None:
    """creates an empty COLMAP database"""
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()


def reconstruct(database_path, image_dir, output_path, mvs_path):
    create_database(database_path)
    add_imgs_to_database(database_path, "/workspaces/BEP-3D-scanner/datasets/ccvpeer_downscaledPY/sparse/images.txt")
    add_cams_to_database(database_path, "/workspaces/BEP-3D-scanner/datasets/ccvpeer_downscaledPY/sparse/cameras.txt")
    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)

    #pycolmap.verify_matches(database_path)
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    print("incremental_mapping done")
    maps[0].write(output_path)
    print("wrote map")
    maps[0].export_PLY(f"{output_path}/sparse_reconstruction.ply")

if __name__ == "__main__":
    database_path = "datasets/ccvpeer_downscaledPY/database.db"
    image_dir = "datasets/ccvpeer_downscaledPY/images/"
    output_path = "datasets/ccvpeer_downscaledPY/"
    mvs_path = output_path + "/mvs/"
    #database_path = "datasets/lego/train2/database.db"
    #image_dir = "datasets/lego/train2"
    reconstruct(database_path, image_dir, output_path, mvs_path)

