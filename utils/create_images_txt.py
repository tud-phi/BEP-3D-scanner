import os
from spherical_to_quaternion import compute_pose

print(os.listdir("datasets/ignore_plastic_pear/images"))

path = "datasets/ignore_plastic_pear/images"

img_names = os.listdir(path)
# takes names in this format: "test_theta_phi.jpg"
spherical_coordinates = [img_name.split(".")[0].split("_")[1:] for img_name in img_names]
print(spherical_coordinates)
radius = 320
with open('images.txt', 'w') as f:
    f.write('# handmade images.txt file!')

    for i, image_name in enumerate(img_names):
        theta, phi = spherical_coordinates[i][0], spherical_coordinates[i][1]
        pose = compute_pose(radius, float(theta), float(phi))
        f.write('\n')
        f.write(str(i)+" ")
        f.write(" ".join(str(x) for x in pose))
        f.write(" 1 ")
        f.write(img_names[i])
        f.write('\n')