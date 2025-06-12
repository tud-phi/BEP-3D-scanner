import point_cloud_utils as pcu
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

# Note: Adapted from the example script at https://fwilliams.info/point-cloud-utils/sections/closest_point_on_mesh/


# Load the mesh to measure the scan against:
#   v is a NumPy array of coordinates with shape (V, 3)
#   f is a NumPy array of face indices with shape (F, 3)
# v_full, f_full = pcu.load_mesh_vf("../../datasets/Stanford_Bunny/stanford-bunny.obj") # More robust path
v_full, f_full = pcu.load_mesh_vf("../../datasets/Stanford_Bunny/stanford-bunny.obj") # More robust path

# Downsample the Bunny mesh to make it faster to run
print("Number of faces in original mesh: ", f_full.shape[0])

face_reduction = 0.005 # Reduce the mesh by a %
num_faces = math.ceil(f_full.shape[0] * face_reduction)

v, f, corr_qv, corr_qf = pcu.decimate_triangle_mesh(v_full, f_full, max_faces=num_faces)


print("Number of faces in downsampled mesh: ", f.shape[0])
        
# Generate random points on a sphere around the shape

num_points = 1000
radius_scale = 1.0 
center = v.mean(axis=0)
radius = np.max(np.linalg.norm(v - center, axis=1)) * radius_scale # Scale radius based on bunny size
p = np.random.randn(num_points, 3)
p_norm = np.linalg.norm(p, axis=-1, keepdims=True)
p = center + (p / p_norm) * radius # Points on a sphere centered around the mesh

# Compute the shortest distance between each point in p and the mesh:
#   dists is a NumPy array of shape (P,) where dists[i] is the
#   shortest distnace between the point p[i, :] and the mesh (v, f)
dists, fid, bc = pcu.closest_points_on_mesh(p, v, f)

# Interpolate the barycentric coordinates to get the coordinates of
# the closest points on the mesh to each point in p
closest_pts = pcu.interpolate_barycentric_coords(f, fid, bc, v)

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the original mesh
ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color='lightgray', alpha=1.0, edgecolor='gray', linewidth=0.1)

# Plot the new point cloud, colored by distance to the surface mesh
# Normalize distances for color mapping
norm = plt.Normalize(dists.min(), dists.max())
# cmap = cm.viridis 
cmap = cm.cividis 

scatter_points = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=dists, cmap=cmap, s=50, label='Query Points', alpha=0.1)

# Draw lines between the points and their closest points on the mesh
for i in range(p.shape[0]):
    ax.plot([p[i, 0], closest_pts[i, 0]],
            [p[i, 1], closest_pts[i, 1]],
            [p[i, 2], closest_pts[i, 2]],
            'k-', lw=0.8, alpha=0.1) # Black lines

# Add a colorbar
cbar = fig.colorbar(scatter_points, ax=ax, shrink=0.7, aspect=10)
cbar.set_label('Distance to Surface')

# Set labels and title
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.set_zlabel("Z coordinate")
ax.set_title("Mesh, Query Points, and Closest Point Connections")

# Adjust plot limits for better visualization
# Combine all points to find encompassing bounds
all_points = np.vstack((v, p, closest_pts))
min_coords = all_points.min(axis=0)
max_coords = all_points.max(axis=0)
mid_coords = (min_coords + max_coords) / 2
range_coords = (max_coords - min_coords).max() / 1.5 # Adjust zoom level

ax.set_xlim(mid_coords[0] - range_coords, mid_coords[0] + range_coords)
ax.set_ylim(mid_coords[1] - range_coords, mid_coords[1] + range_coords)
ax.set_zlim(mid_coords[2] - range_coords, mid_coords[2] + range_coords)


ax.legend()
plt.tight_layout()
plt.show()

print("Distances:", dists)
print("Closest points on mesh:", closest_pts)
