#!/usr/bin/env python3
"""
Pose Deformation System
This module applies joint transformations to deform skinned meshes in real-time.
"""

import numpy as np
import trimesh
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from armature_utils import BoneInfo


@dataclass
class BoneTransform:
    """Transformation data for a bone."""
    bone_name: str
    translation: np.ndarray = None  # 3D translation
    rotation: np.ndarray = None     # 3x3 rotation matrix
    full_transform: np.ndarray = None  # 4x4 transformation matrix
    
    def __post_init__(self):
        if self.translation is None:
            self.translation = np.zeros(3)
        if self.rotation is None:
            self.rotation = np.eye(3)
        if self.full_transform is None:
            self.full_transform = np.eye(4)


class PoseDeformer:
    """Applies pose transformations to skinned meshes."""
    
    def __init__(self, bones: Dict[str, BoneInfo]):
        self.bones = bones
        self.base_bone_transforms = {}
        self._compute_base_transforms()
    
    def _compute_base_transforms(self):
        """Compute base transformation matrices for all bones."""
        print("Computing base bone transformations...")
        
        for bone_name, bone_info in self.bones.items():
            # Create base transformation matrix (bone in rest pose)
            bone_vector = bone_info.tail - bone_info.head
            bone_length = np.linalg.norm(bone_vector)
            
            transform = np.eye(4)
            transform[:3, 3] = bone_info.head  # Translation to bone head
            
            # Store base transform
            self.base_bone_transforms[bone_name] = BoneTransform(
                bone_name=bone_name,
                translation=bone_info.head,
                rotation=np.eye(3),
                full_transform=transform
            )
            
            print(f"  {bone_name}: head={bone_info.head}, length={bone_length:.3f}")
        
        print(f"Computed {len(self.base_bone_transforms)} base transforms")
    
    def apply_pose(self, pose_angles: Dict[str, np.ndarray]) -> Dict[str, BoneTransform]:
        """
        Apply pose angles to bones and compute final transformations.
        
        Args:
            pose_angles: Dictionary mapping bone_name to rotation angles (in radians)
            
        Returns:
            Dictionary of final bone transformations with proper bind pose handling
        """
        print(f"\n=== Applying Pose ===")
        print(f"Pose contains {len(pose_angles)} joint modifications:")
        for bone_name, angles in pose_angles.items():
            print(f"  {bone_name}: {np.degrees(angles)} degrees")
        
        # Create transforms that represent the CHANGE from rest pose
        final_transforms = {}
        for bone_name, base_transform in self.base_bone_transforms.items():
            # Start with identity transform (no change from rest pose)
            final_transforms[bone_name] = BoneTransform(
                bone_name=bone_name,
                translation=np.zeros(3),  # No translation change
                rotation=np.eye(3),       # No rotation change
                full_transform=np.eye(4)  # Identity = no change
            )
        
        # Apply ONLY the pose modifications (changes from rest)
        transforms_applied = 0
        for bone_name, angles in pose_angles.items():
            if bone_name in final_transforms and bone_name in self.bones:
                # Convert angles to rotation matrix
                rotation = Rotation.from_euler('xyz', angles)
                rotation_matrix = rotation.as_matrix()
                
                bone_info = self.bones[bone_name]
                joint_position = bone_info.head
                
                # Create transformation that rotates around the joint position
                # This is the key fix: rotate around joint, not origin
                transform = np.eye(4)
                
                # Step 1: Translate to joint
                translate_to_joint = np.eye(4)
                translate_to_joint[:3, 3] = -joint_position
                
                # Step 2: Apply rotation
                rotate = np.eye(4)
                rotate[:3, :3] = rotation_matrix
                
                # Step 3: Translate back
                translate_back = np.eye(4)
                translate_back[:3, 3] = joint_position
                
                # Combine: translate back * rotate * translate to joint
                transform = translate_back @ rotate @ translate_to_joint
                
                # Update the transform
                bone_transform = final_transforms[bone_name]
                bone_transform.rotation = rotation_matrix
                bone_transform.full_transform = transform
                
                transforms_applied += 1
                print(f"  ✅ Applied {np.linalg.norm(angles)*180/np.pi:.1f}° rotation to {bone_name} around joint at {joint_position}")
        
        print(f"Applied {transforms_applied} joint rotations")
        return final_transforms
    
    def _propagate_transform(self, parent_bone: str, parent_transform: np.ndarray, 
                           all_transforms: Dict[str, BoneTransform]):
        """Propagate parent transformation to child bones."""
        # Find child bones
        parent_bone_info = self.bones[parent_bone]
        
        # For now, simple propagation - in a full system this would be more complex
        # This is a simplified version to show the concept
        pass
    
    def deform_mesh(self, mesh: trimesh.Trimesh, vertex_weights: List[List[Tuple[str, float]]], 
                   bone_transforms: Dict[str, BoneTransform]) -> trimesh.Trimesh:
        """
        Deform a mesh based on bone transformations and vertex weights.
        
        Args:
            mesh: The mesh to deform
            vertex_weights: Per-vertex bone weights [(bone_name, weight), ...]
            bone_transforms: Current bone transformations
            
        Returns:
            Deformed mesh
        """
        print(f"\n=== Deforming Mesh ===")
        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Vertex weights: {len(vertex_weights)} entries")
        
        # Create a copy of the mesh
        deformed_mesh = mesh.copy()
        original_vertices = mesh.vertices.copy()
        deformed_vertices = np.zeros_like(original_vertices)
        
        vertices_affected = 0
        total_displacement = 0.0
        max_displacement = 0.0
        
        # Deform each vertex
        for i, vertex_bone_weights in enumerate(vertex_weights):
            if not vertex_bone_weights:
                # No weights, keep original position
                deformed_vertices[i] = original_vertices[i]
                continue
            
            original_vertex = original_vertices[i]
            
            # Check if any bones affecting this vertex actually have transformations
            has_active_bones = False
            for bone_name, weight in vertex_bone_weights:
                if (bone_name in bone_transforms and weight > 0.001 and 
                    not np.allclose(bone_transforms[bone_name].full_transform, np.eye(4))):
                    has_active_bones = True
                    break
            
            if not has_active_bones:
                # No active transformations, keep original position
                deformed_vertices[i] = original_vertices[i]
                continue
            
            # Apply weighted transformation from all influencing bones
            new_vertex = original_vertex.copy()  # Start with original position
            
            for bone_name, weight in vertex_bone_weights:
                if bone_name in bone_transforms and weight > 0.001:
                    bone_transform = bone_transforms[bone_name]
                    
                    # Only apply transform if it's not identity
                    if not np.allclose(bone_transform.full_transform, np.eye(4)):
                        # Transform vertex by this bone
                        homogeneous_vertex = np.append(original_vertex, 1.0)
                        transformed_vertex = bone_transform.full_transform @ homogeneous_vertex
                        
                        # Apply weighted transformation (blend between original and transformed)
                        contribution = weight * (transformed_vertex[:3] - original_vertex)
                        new_vertex += contribution
            
            # Calculate displacement
            displacement = np.linalg.norm(new_vertex - original_vertex)
            total_displacement += displacement
            max_displacement = max(max_displacement, displacement)
            
            if displacement > 0.001:  # Only count significantly moved vertices
                vertices_affected += 1
            
            deformed_vertices[i] = new_vertex
        
        # Update mesh vertices
        deformed_mesh.vertices = deformed_vertices
        
        # Store deformation statistics in metadata for testing
        avg_displacement = total_displacement / vertices_affected if vertices_affected > 0 else 0
        deformed_mesh.metadata['deformation_stats'] = {
            'total_vertices': len(vertex_weights),
            'vertices_moved': vertices_affected,
            'vertices_unchanged': len(vertex_weights) - vertices_affected,
            'avg_displacement': avg_displacement,
            'max_displacement': max_displacement,
            'total_displacement': total_displacement
        }
        
        # Log deformation statistics
        print(f"Deformation results:")
        print(f"  Total vertices: {len(vertex_weights)}")
        print(f"  Vertices moved: {vertices_affected} ({100*vertices_affected/len(vertex_weights):.1f}%)")
        print(f"  Vertices unchanged: {len(vertex_weights)-vertices_affected}")
        
        if vertices_affected > 0:
            print(f"  Average displacement (moved vertices): {avg_displacement:.4f} units")
            print(f"  Maximum displacement: {max_displacement:.4f} units")
            
            if max_displacement < 0.001:
                print("  ⚠️  WARNING: Deformation is very small - might not be visible")
            elif max_displacement > 0.5:
                print("  ⚠️  WARNING: Very large deformation - may look distorted")
            else:
                print(f"  ✅ Reasonable deformation range")
        else:
            print("  ℹ️  No vertices moved (no active bone transforms)")
        
        return deformed_mesh
    
    def create_posed_scene(self, base_scene: trimesh.Scene, pose_angles: Dict[str, np.ndarray]) -> trimesh.Scene:
        """
        Create a new scene with all meshes deformed according to the pose.
        
        Args:
            base_scene: Original scene with skinned meshes
            pose_angles: Joint angles to apply
            
        Returns:
            New scene with deformed meshes
        """
        print(f"\n=== Creating Posed Scene ===")
        print(f"Base scene: {len(base_scene.geometry)} geometries")
        
        # Apply pose to bones
        bone_transforms = self.apply_pose(pose_angles)
        
        # Create new scene
        posed_scene = trimesh.Scene()
        meshes_processed = 0
        meshes_deformed = 0
        
        # Process each mesh in the scene
        for node_name, geometry in base_scene.geometry.items():
            if hasattr(geometry, 'metadata') and 'skinning' in geometry.metadata:
                # This mesh has skinning data
                skinning_data = geometry.metadata['skinning']
                vertex_weights = skinning_data['vertex_weights']
                
                print(f"\nProcessing skinned mesh: {node_name}")
                print(f"  Body: {skinning_data['body_name']}")
                print(f"  Mesh: {skinning_data['mesh_name']}")
                
                # Deform the mesh
                deformed_mesh = self.deform_mesh(geometry, vertex_weights, bone_transforms)
                posed_scene.add_geometry(deformed_mesh, node_name=node_name)
                
                meshes_processed += 1
                # Check if mesh was actually deformed
                original_center = np.mean(geometry.vertices, axis=0)
                deformed_center = np.mean(deformed_mesh.vertices, axis=0)
                center_displacement = np.linalg.norm(deformed_center - original_center)
                
                if center_displacement > 0.001:
                    meshes_deformed += 1
                    print(f"  ✅ Mesh center moved by {center_displacement:.6f} units")
                else:
                    print(f"  ⚠️  Mesh center unchanged")
                
            else:
                # Non-skinned mesh, add as-is
                posed_scene.add_geometry(geometry, node_name=node_name)
                print(f"Added non-skinned geometry: {node_name}")
        
        print(f"\n=== Scene Creation Complete ===")
        print(f"Processed {meshes_processed} skinned meshes")
        print(f"Successfully deformed {meshes_deformed} meshes")
        print(f"Total geometries in posed scene: {len(posed_scene.geometry)}")
        
        if meshes_deformed == 0:
            print("⚠️  WARNING: No meshes were visibly deformed!")
        
        return posed_scene


def main():
    """Test the pose deformation system."""
    print("Pose Deformation System - Test Mode")
    print("This module requires bones and skinned meshes to test properly.")


if __name__ == "__main__":
    main()
