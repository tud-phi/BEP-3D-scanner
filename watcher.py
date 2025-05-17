import time
import os
from colmap.reconstruct import reconstruct_unknown_poses
ready = os.path.isfile("datasets/docker_test/READY.txt")

database_path = "datasets/docker_test/database.db"
image_dir = "datasets/docker_test/images/"
output_path = "datasets/docker_test/"
known_parameters_path = "/workspaces/BEP-3D-scanner/datasets/docker_test/known_parameters/"

while not ready:
    ready = os.path.isfile("datasets/docker_test/READY.txt")
    if ready:
        reconstruct_unknown_poses(database_path, image_dir, output_path)
        os.mkdir("datasets/docker_test/klaar2")
    time.sleep(1)
