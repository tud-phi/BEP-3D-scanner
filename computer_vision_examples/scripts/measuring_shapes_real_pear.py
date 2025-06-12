import point_cloud_utils as pcu
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

def add_gaussian_noise_to_mesh(vertices, faces, mean=0.0, std=0.1):
    """
    Adds Gaussian noise to the vertices of a mesh.

    Args:
        vertices (np.ndarray): A NumPy array of shape (N, 3) representing
                               N vertices in 3D space.
        faces (np.ndarray): A NumPy array of shape (M, K) representing M faces,
                            where K is the number of vertices per face. This
                            array contains indices into the vertices array.
        mean (float): The mean of the Gaussian distribution for the noise.
        std (float): The standard deviation (spread) of the Gaussian
                     distribution for the noise.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - noisy_vertices (np.ndarray): The vertices with added noise.
            - faces (np.ndarray): The original faces array, which remains
                                  unchanged as it only contains indices.
    """
    # Generate Gaussian noise with the same shape as the vertices array
    # This creates a unique noise vector for each vertex
    noise = np.random.normal(mean, std, vertices.shape)
    
    # Add the generated noise to the original vertex positions
    noisy_vertices = vertices + noise
    
    # The faces array does not need to be modified because it just contains
    # indices to the vertices. The connectivity remains the same.
    return noisy_vertices, faces

def scale_mesh(vertices, faces, scale_factor=1.5):
    """
    Scales a mesh by a given factor around its geometric center.

    Args:
        vertices (np.ndarray): A NumPy array of shape (N, 3) representing
                               N vertices in 3D space.
        faces (np.ndarray): A NumPy array of shape (M, K) representing M faces.
                            This is not modified but returned for consistency.
        scale_factor (float): The factor by which to scale the mesh.
                              > 1.0 to enlarge, < 1.0 to shrink.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - scaled_vertices (np.ndarray): The vertices of the scaled mesh.
            - faces (np.ndarray): The original faces array.
    """
    # Calculate the geometric center (centroid) of the mesh
    centroid = np.mean(vertices, axis=0)
    
    # Translate vertices to the origin
    vertices_centered = vertices - centroid
    
    # Scale the centered vertices
    scaled_centered_vertices = vertices_centered * scale_factor
    
    # Translate the scaled vertices back to the original centroid position
    scaled_vertices = scaled_centered_vertices + centroid
    
    return scaled_vertices, faces


# Load the mesh to measure the scan against:
#   v is a NumPy array of coordinates with shape (V, 3)
#   f is a NumPy array of face indices with shape (F, 3)
# Note: change this to the stl file of the pear
v_full, f_full = pcu.load_mesh_vf("../../datasets/Stanford_Bunny/stanford-bunny.obj") # More robust path



# Note this section can just be replaced with the test shape
# --------------------------------------------------------- #
# Downsample mesh
face_reduction = 0.005 # Reduce the mesh by a %
num_faces = math.ceil(f_full.shape[0] * face_reduction)

v, f, corr_qv, corr_qf = pcu.decimate_triangle_mesh(v_full, f_full, max_faces=num_faces)

# --- calculate normals for the mesh --- #
n = pcu.estimate_mesh_vertex_normals(v, f)

# --- Scale the mesh of the stanford bunny --- #
v_scaled, f_scaled = scale_mesh(v, f, scale_factor=1.1)

# ------------------------------------------------------ #

# --- resample the mesh --- # 
# Generate barycentric coordinates of random samples
num_samples = 2000
fid, bc = pcu.sample_mesh_random(v_scaled, f_scaled, num_samples)

# Interpolate the vertex positions and normals using the returned barycentric coordinates
# to get sample positions and normals
rand_positions = pcu.interpolate_barycentric_coords(f, fid, bc, v)
rand_normals = pcu.interpolate_barycentric_coords(f, fid, bc, n)


# Add noise to the mesh
noisy_points, noisy_faces = add_gaussian_noise_to_mesh(rand_positions, f, mean=0.0, std=0.01)



# Compute the shortest distance between each point in p and the mesh:
#   dists is a NumPy array of shape (P,) where dists[i] is the
#   shortest distnace between the point p[i, :] and the mesh (v, f)
dists, fid, bc = pcu.closest_points_on_mesh(noisy_points, v, f)

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

scatter_points = ax.scatter(noisy_points[:, 0], noisy_points[:, 1], noisy_points[:, 2], c=dists, cmap=cmap, s=10, label='Query Points', alpha=0.4)

# Draw lines between the points and their closest points on the mesh
for i in range(noisy_points.shape[0]):
    ax.plot([noisy_points[i, 0], closest_pts[i, 0]],
            [noisy_points[i, 1], closest_pts[i, 1]],
            [noisy_points[i, 2], closest_pts[i, 2]],
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
all_points = np.vstack((v, noisy_points, closest_pts))
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

# print("Distances:", dists)
# print("Closest points on mesh:", closest_pts)

# Visualize the errors using a violen plot

# --- Create the second violin plot (showing means) ---


fig2, ax2 = plt.subplots(figsize=(10, 6)) # Adjust figure size as needed

# Create the violin plot, this time showing means
parts2 = ax2.violinplot(dists, showmeans=True, showmedians=False, showextrema=True)
# parts2 = ax2.violinplot(datasets, showmeans=True, showmedians=False, showextrema=False)

# Style the violins for the second plot
for pc in parts2['bodies']:
    pc.set_facecolor('#A8DADC') # A light teal/blue color
    pc.set_edgecolor('black')
    pc.set_alpha(0.8)

# Style means and other parts for the second plot
parts2['cmeans'].set_edgecolor('blue') # Style for means
parts2['cmeans'].set_linewidth(2)
parts2['cbars'].set_edgecolor('black')
parts2['cmaxes'].set_edgecolor('black')
parts2['cmins'].set_edgecolor('black')


# Set plot title and labels for the second plot
ax2.set_title('Comparision of Pear reconstruction vs. CAD mesh', fontsize=16)
ax2.set_xlabel('Pear model', fontsize=14)
ax2.set_ylabel('Accuracy (m)', fontsize=14)

# Set x-axis ticks and labels for the second plot
# ax2.set_xticks(np.arange(1, len(labels) + 1))
# ax2.set_xticklabels(labels, rotation=45, ha="right")

# Add a grid for better readability of y-values for the second plot
ax2.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

# Ensure layout is tight for the second plot
# fig2.tight_layout()

plt.show()

