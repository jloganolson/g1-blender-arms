#!/usr/bin/env python3
"""
Armature Utilities for MJCF to GLB conversion
This module creates Blender armatures from MJCF joint hierarchies.
"""

import numpy as np
import trimesh
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from mjcf_parser import MJCFParser, JointInfo, BodyInfo


@dataclass
class BoneInfo:
    """Information about a bone in the armature."""
    name: str
    head: np.ndarray  # 3D position of bone head (joint position)
    tail: np.ndarray  # 3D position of bone tail (child joint position or extrapolated)
    parent: Optional[str] = None
    joint_info: Optional[JointInfo] = None
    body_name: Optional[str] = None


class ArmatureBuilder:
    """Builds armatures from MJCF joint hierarchies."""
    
    def __init__(self, parser: MJCFParser):
        self.parser = parser
        self.bones: Dict[str, BoneInfo] = {}
    
    def compute_bone_positions(self) -> Dict[str, BoneInfo]:
        """
        Compute bone positions from the MJCF joint hierarchy.
        Each bone represents a joint and extends toward its child joint.
        """
        bones = {}
        
        # First pass: create bones for each joint at their body positions
        for body_name, body_info in self.parser.bodies.items():
            if body_info.joint:
                joint_name = body_info.joint.name
                
                # Compute global position for this body (joint position)
                global_pos, _ = self.parser.compute_global_transform(body_name)
                
                # Find parent joint (walk up the hierarchy until we find a body with a joint)
                parent_joint_name = None
                current_parent = body_info.parent
                while current_parent and current_parent in self.parser.bodies:
                    parent_body = self.parser.bodies[current_parent]
                    if parent_body.joint:
                        parent_joint_name = parent_body.joint.name
                        break
                    current_parent = parent_body.parent
                
                # Create bone with head at joint position
                bones[joint_name] = BoneInfo(
                    name=joint_name,
                    head=global_pos.copy(),
                    tail=global_pos.copy() + np.array([0, 0, 0.1]),  # Default tail offset
                    parent=parent_joint_name,
                    joint_info=body_info.joint,
                    body_name=body_name
                )
        
        # Second pass: set proper tail positions based on children
        for joint_name, bone in bones.items():
            children = self._find_child_joints(joint_name, bones)
            
            if children:
                # Point toward the first child (or average of children)
                if len(children) == 1:
                    child_pos = bones[children[0]].head.copy()
                    bone_vector = child_pos - bone.head
                    bone_length = np.linalg.norm(bone_vector)
                    
                    # Limit bone length to reasonable size (max 8cm for limb bones)
                    max_length = 0.08
                    if bone_length > max_length:
                        bone_vector = (bone_vector / bone_length) * max_length
                    
                    bone.tail = bone.head + bone_vector
                else:
                    # Average position of all children
                    child_positions = [bones[child].head for child in children]
                    avg_pos = np.mean(child_positions, axis=0)
                    bone_vector = avg_pos - bone.head
                    bone_length = np.linalg.norm(bone_vector)
                    
                    # Limit bone length
                    max_length = 0.08
                    if bone_length > max_length:
                        bone_vector = (bone_vector / bone_length) * max_length
                    
                    bone.tail = bone.head + bone_vector
            else:
                # Leaf bone: make it much shorter and point inward toward parent
                if bone.parent and bone.parent in bones:
                    # Point back toward parent (inward)
                    parent_bone = bones[bone.parent]
                    direction = (parent_bone.head - bone.head)
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 0.001:
                        direction = direction / direction_norm
                        bone.tail = bone.head + direction * 0.02  # 2cm short inward bone
                    else:
                        direction = self._get_bone_direction(bone)
                        bone.tail = bone.head + direction * 0.02  # 2cm default length
                else:
                    # Root bone or no parent, use default but shorter
                    direction = self._get_bone_direction(bone)
                    bone.tail = bone.head + direction * 0.02  # 2cm default length
        
        self.bones = bones
        return bones
    
    def _find_child_joints(self, joint_name: str, bones: Dict[str, BoneInfo]) -> List[str]:
        """Find all child joints of a given joint."""
        children = []
        for bone_name, bone in bones.items():
            if bone.parent == joint_name:
                children.append(bone_name)
        return children
    
    def _get_bone_direction(self, bone: BoneInfo) -> np.ndarray:
        """Get the direction a bone should point based on its joint axis."""
        if bone.joint_info and bone.joint_info.axis:
            # Use joint axis as direction hint
            axis = np.array(bone.joint_info.axis)
            # Cross product with up vector to get a perpendicular direction
            up = np.array([0, 0, 1])
            direction = np.cross(axis, up)
            # If parallel to up, use a different reference
            if np.linalg.norm(direction) < 0.1:
                direction = np.cross(axis, np.array([1, 0, 0]))
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            return direction
        else:
            # Default direction (down)
            return np.array([0, 0, -1])
    
    def create_trimesh_armature(self) -> Optional[trimesh.Scene]:
        """
        Create a visual representation of the armature using trimesh.
        Returns a scene with cylinder bones for visualization.
        """
        if not self.bones:
            self.compute_bone_positions()
        
        scene = trimesh.Scene()
        
        for bone_name, bone in self.bones.items():
            # Create a cylinder to represent the bone
            bone_vector = bone.tail - bone.head
            bone_length = np.linalg.norm(bone_vector)
            
            if bone_length < 0.001:  # Skip very short bones
                continue
            
            # Create cylinder
            cylinder = trimesh.creation.cylinder(
                radius=0.01,  # 1cm radius
                height=bone_length,
                sections=8
            )
            
            # Transform cylinder to align with bone
            bone_direction = bone_vector / bone_length
            
            # Create transformation matrix
            transform = np.eye(4)
            
            # Position at bone center
            bone_center = (bone.head + bone.tail) / 2
            transform[:3, 3] = bone_center
            
            # Align with bone direction (cylinder default is along Z)
            z_axis = np.array([0, 0, 1])
            if not np.allclose(bone_direction, z_axis):
                rotation_axis = np.cross(z_axis, bone_direction)
                rotation_angle = np.arccos(np.clip(np.dot(z_axis, bone_direction), -1, 1))
                if np.linalg.norm(rotation_axis) > 0.001:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    rotation = Rotation.from_rotvec(rotation_angle * rotation_axis)
                    transform[:3, :3] = rotation.as_matrix()
            
            # Apply transformation
            cylinder.apply_transform(transform)
            
            # Add to scene with bone name
            scene.add_geometry(cylinder, node_name=f"bone_{bone_name}")
        
        return scene
    
    def get_bone_hierarchy_for_export(self) -> Dict[str, Dict[str, Any]]:
        """
        Get bone hierarchy in a format suitable for GLB export.
        This will be used for creating the actual armature in the GLB file.
        """
        if not self.bones:
            self.compute_bone_positions()
        
        hierarchy = {}
        for bone_name, bone in self.bones.items():
            hierarchy[bone_name] = {
                'head': bone.head.tolist(),
                'tail': bone.tail.tolist(),
                'parent': bone.parent,
                'joint_type': bone.joint_info.type if bone.joint_info else 'unknown',
                'joint_axis': bone.joint_info.axis if bone.joint_info and bone.joint_info.axis else None,
                'joint_range': bone.joint_info.range if bone.joint_info and bone.joint_info.range else None,
                'body_name': bone.body_name
            }
        return hierarchy
    
    def print_armature_info(self):
        """Print armature information for debugging."""
        if not self.bones:
            self.compute_bone_positions()
        
        print("=== Armature Information ===")
        print(f"Total bones: {len(self.bones)}")
        
        # Find root bones
        root_bones = [name for name, bone in self.bones.items() if bone.parent is None]
        print(f"Root bones: {root_bones}")
        
        # Print hierarchy
        for root in root_bones:
            self._print_bone_hierarchy(root, 0)
    
    def _print_bone_hierarchy(self, bone_name: str, indent: int):
        """Print bone hierarchy recursively."""
        if bone_name not in self.bones:
            return
        
        bone = self.bones[bone_name]
        joint_type = bone.joint_info.type if bone.joint_info else "unknown"
        bone_length = np.linalg.norm(bone.tail - bone.head)
        
        print("  " * indent + f"{bone_name} ({joint_type}): "
              f"head={bone.head}, length={bone_length:.3f}")
        
        # Print children
        children = self._find_child_joints(bone_name, self.bones)
        for child in children:
            self._print_bone_hierarchy(child, indent + 1)


def main():
    """Test the armature builder."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python armature_utils.py <mjcf_file>")
        return
    
    mjcf_file = sys.argv[1]
    print(f"Building armature from {mjcf_file}")
    
    # Parse MJCF
    parser = MJCFParser(mjcf_file)
    
    # Build armature
    armature_builder = ArmatureBuilder(parser)
    armature_builder.print_armature_info()
    
    # Create visual representation
    armature_scene = armature_builder.create_trimesh_armature()
    if armature_scene:
        # Export armature visualization
        output_path = "test_armature.glb"
        armature_scene.export(output_path)
        print(f"\nArmature visualization exported to: {output_path}")


if __name__ == "__main__":
    main()
