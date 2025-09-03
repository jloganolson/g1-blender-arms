#!/usr/bin/env python3
"""
STL to GLB Mesh Simplifier
This script loads an STL file, reduces its polycount, exports to GLB, and renders a preview.
"""

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def simplify_mesh(mesh, target_faces=None, reduction_ratio=0.5):
    """
    Simplify a mesh by reducing the number of faces.
    
    Args:
        mesh: trimesh.Trimesh object
        target_faces: Target number of faces (optional)
        reduction_ratio: Ratio to reduce faces by (default 0.5 = 50% reduction)
    
    Returns:
        Simplified trimesh.Trimesh object
    """
    print(f"Original mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    
    if target_faces is not None:
        # Calculate reduction ratio from target faces
        reduction_ratio = target_faces / len(mesh.faces)
        if reduction_ratio > 1.0:
            reduction_ratio = 1.0
            print(f"Warning: Target faces ({target_faces}) is greater than original faces ({len(mesh.faces)}). Using original mesh.")
        elif reduction_ratio < 0.01:
            reduction_ratio = 0.01
            print(f"Warning: Target faces too low. Using minimum reduction ratio of 0.01")
    
    print(f"Reduction ratio: {reduction_ratio:.3f}")
    target_faces_calc = int(len(mesh.faces) * reduction_ratio)
    print(f"Target faces: {target_faces_calc}")
    
    # Use trimesh's built-in simplification with face count
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces_calc)
    
    print(f"Simplified mesh: {len(simplified.faces)} faces, {len(simplified.vertices)} vertices")
    
    return simplified


def export_to_glb(mesh, output_path):
    """
    Export mesh to GLB format.
    
    Args:
        mesh: trimesh.Trimesh object
        output_path: Path to save GLB file
    """
    # Create a scene with the mesh
    scene = trimesh.Scene([mesh])
    
    # Export to GLB
    exported = scene.export(file_type='glb')
    
    with open(output_path, 'wb') as f:
        f.write(exported)
    
    print(f"Exported to: {output_path}")


def render_mesh(mesh, title="Mesh Preview", save_path=None):
    """
    Render a mesh using matplotlib.
    
    Args:
        mesh: trimesh.Trimesh object
        title: Title for the plot
        save_path: Path to save the plot (optional)
    """
    # Set matplotlib to use a non-interactive backend
    import matplotlib
    matplotlib.use('Agg')  # Use Anti-Grain Geometry backend for non-interactive plotting
    
    # Create a figure with 3D projection
    fig = plt.figure(figsize=(12, 5))
    
    # Original mesh wireframe
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_trisurf(mesh.vertices[:, 0], 
                     mesh.vertices[:, 1], 
                     mesh.vertices[:, 2], 
                     triangles=mesh.faces,
                     alpha=0.8,
                     cmap='viridis')
    ax1.set_title(f'{title} - Wireframe')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Solid rendering
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_trisurf(mesh.vertices[:, 0], 
                     mesh.vertices[:, 1], 
                     mesh.vertices[:, 2], 
                     triangles=mesh.faces,
                     alpha=0.9,
                     cmap='plasma',
                     shade=True)
    ax2.set_title(f'{title} - Solid')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # Generate default filename
        safe_title = title.replace(" ", "_").lower()
        save_path = f"{safe_title}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close(fig)  # Close figure to free memory


def main():
    parser = argparse.ArgumentParser(description='Simplify STL mesh and export to GLB')
    parser.add_argument('input_stl', help='Path to input STL file')
    parser.add_argument('-o', '--output', help='Output GLB file path (default: input_name_simplified.glb)')
    parser.add_argument('-r', '--ratio', type=float, default=0.5, 
                       help='Reduction ratio (0.1 = 90%% reduction, 0.5 = 50%% reduction)')
    parser.add_argument('-t', '--target-faces', type=int, 
                       help='Target number of faces (overrides ratio)')
    parser.add_argument('--no-render', action='store_true', 
                       help='Skip rendering preview')
    
    args = parser.parse_args()
    
    # Load the STL file
    input_path = Path(args.input_stl)
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist!")
        return
    
    print(f"Loading STL file: {input_path}")
    mesh = trimesh.load(str(input_path))
    
    # Ensure it's a mesh (not a scene)
    if isinstance(mesh, trimesh.Scene):
        # If it's a scene, get the first mesh
        mesh = list(mesh.geometry.values())[0]
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_simplified.glb"
    
    # Render original mesh if requested
    if not args.no_render:
        print("Rendering original mesh...")
        render_mesh(mesh, "Original Mesh")
    
    # Simplify the mesh
    simplified_mesh = simplify_mesh(mesh, args.target_faces, args.ratio)
    
    # Export to GLB
    export_to_glb(simplified_mesh, output_path)
    
    # Render simplified mesh if requested
    if not args.no_render:
        print("Rendering simplified mesh...")
        render_mesh(simplified_mesh, "Simplified Mesh")
    
    print(f"\nProcess completed!")
    print(f"Original: {len(mesh.faces)} faces")
    print(f"Simplified: {len(simplified_mesh.faces)} faces")
    print(f"Reduction: {((len(mesh.faces) - len(simplified_mesh.faces)) / len(mesh.faces) * 100):.1f}%")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    # If run without arguments, use the pelvis.STL as default
    import sys
    if len(sys.argv) == 1:
        # Default test with pelvis.STL
        default_stl = "./g1_description/meshes/pelvis.STL"
        sys.argv.extend([default_stl, "--ratio", "0.3"])
    
    main()
