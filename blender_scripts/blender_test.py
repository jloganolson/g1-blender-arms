#blender -b -P blender_scripts/blender_test.py
import bpy

# Create a cube
bpy.ops.mesh.primitive_cube_add(size=2)

# Set render output path
bpy.context.scene.render.filepath = "./output.png"

# Render an image
bpy.ops.render.render(write_still=True)
