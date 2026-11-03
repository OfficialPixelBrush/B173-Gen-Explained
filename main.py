import sys
import os
import importlib

import bpy
import bmesh
import numpy as np

# --- Settings ---------------------------------------------------------------

# Folder containing chunk_gen.py, perlin_gen.py, java_random.py, java_math.py
PROJECT_DIR = r"/home/torben/GitHub/Blender/0016 B173 Gen/"

SEED = 404
CHUNK_X = 0
CHUNK_Z = 0

CELLS = 4                   # noise cells per chunk horizontally
SAMPLES = CELLS + 1         # 5 samples per horizontal axis
LAYERS = 17                 # 17 vertical samples

THRESHOLD = 0.0             # cube exists where density > THRESHOLD
CUBE_SIZE = (1.0, 1.0, 1.0) # (x, y, z) in Blender units per sample
OBJECT_NAME = "DensityChunk"

# --- Import the generator ---------------------------------------------------

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Reload so edits to the generator files are picked up when re-running
for name in ("java_math", "java_random", "perlin_gen", "chunk_gen"):
    if name in sys.modules:
        importlib.reload(sys.modules[name])

from chunk_gen import OverworldGenerator

# --- Generate the density data ----------------------------------------------

gen = OverworldGenerator(SEED)

density = gen.generate_terrain_noise(
    CHUNK_X * CELLS, 0, CHUNK_Z * CELLS,
    SAMPLES, LAYERS, SAMPLES)

# Indexed x * (5 * 17) + z * 17 + y  ->  reshape as (x, z, y)
data = np.array(density, dtype=np.float64).reshape(SAMPLES, SAMPLES, LAYERS)
solid = data > THRESHOLD

print(f"Chunk ({CHUNK_X}, {CHUNK_Z}): {solid.sum()} / {solid.size} solid samples, "
      f"density range {data.min():.3f} .. {data.max():.3f}")

# --- Build the mesh ---------------------------------------------------------

# Each face: (neighbour offset in (x, z, y) array space, 4 corner offsets).
# Array axes are (x, z, y); Blender axes are (X, Y, Z) = (x, z, y).
FACES = [
    ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),  # -X
    (( 1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),  # +X
    ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),  # -Z (Blender -Y)
    ((0,  1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),  # +Z (Blender +Y)
    ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),  # -y (Blender -Z)
    ((0, 0,  1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),  # +y (Blender +Z)
]
# Corner tuples above are (dx, dz, dy)-style offsets in array space; converted below.

def is_solid(x, z, y):
    if 0 <= x < SAMPLES and 0 <= z < SAMPLES and 0 <= y < LAYERS:
        return solid[x, z, y]
    return False

bm = bmesh.new()
vert_cache = {}

def get_vert(x, z, y):
    key = (x, z, y)
    v = vert_cache.get(key)
    if v is None:
        # Array (x, z, y) -> Blender (X, Y, Z)
        v = bm.verts.new((x * CUBE_SIZE[0], z * CUBE_SIZE[1], y * CUBE_SIZE[2]))
        vert_cache[key] = v
    return v

for x in range(SAMPLES):
    for z in range(SAMPLES):
        for y in range(LAYERS):
            if not solid[x, z, y]:
                continue
            for (nx, nz, ny), corners in FACES:
                if is_solid(x + nx, z + nz, y + ny):
                    continue  # hidden face, skip
                verts = [get_vert(x + cx, z + cz, y + cy) for (cx, cz, cy) in corners]
                try:
                    bm.faces.new(verts)
                except ValueError:
                    pass  # face already exists

bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

# --- Put it in the scene ----------------------------------------------------

old = bpy.data.objects.get(OBJECT_NAME)
if old is not None:
    bpy.data.objects.remove(old, do_unlink=True)

mesh = bpy.data.meshes.new(OBJECT_NAME)
bm.to_mesh(mesh)
bm.free()

obj = bpy.data.objects.new(OBJECT_NAME, mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

print(f"Created '{OBJECT_NAME}' with {len(mesh.polygons)} faces")