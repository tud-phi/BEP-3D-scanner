import trimesh

# Load your watertight mesh
mesh = trimesh.load('datasets/peer_constant_f/reconstruct150_surface2.ply')  # or .ply, .stl, etc.

# Compute the volume
volume = mesh.volume

print(f"Mesh volume: {volume:.4f} mm³")