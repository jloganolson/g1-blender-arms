#!/usr/bin/env python3
"""
Skinning Utilities for MJCF to GLB conversion
This module handles vertex weight assignment and mesh rigging to bones.
"""

import numpy as np
import trimesh
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.spatial.distance import cdist

from mjcf_parser import MJCFParser, BodyInfo
from armature_utils import ArmatureBuilder, BoneInfo


@dataclass
class VertexWeight:
    """Vertex weight information for skinning."""
    vertex_index: int
    bone_name: str
    weight: float


@dataclass
class MeshSkinning:
    """Skinning information for a mesh."""
    mesh_name: str
    body_name: str
    vertex_weights: List[List[Tuple[str, float]]]  # For each vertex: [(bone_name, weight), ...]


class SkinnedMeshBuilder:
    """Builds skinned meshes with vertex weights assigned to bones."""
    
    def __init__(self, parser: MJCFParser, armature_builder: ArmatureBuilder):
        self.parser = parser
        self.armature_builder = armature_builder
        self.bones = armature_builder.bones
        if not self.bones:
            self.bones = armature_builder.compute_bone_positions()
        
        self.mesh_skinning: Dict[str, MeshSkinning] = {}
    
    def assign_vertex_weights(self, mesh: trimesh.Trimesh, body_name: str, mesh_name: str) -> MeshSkinning:
        """
        Assign vertex weights for a mesh based on distance to bones.
        Each vertex gets weighted to the closest bones in the kinematic chain.
        """
        vertices = mesh.vertices
        vertex_weights = []
        
        # Find the primary bone for this body
        primary_bone = None
        body_info = self.parser.bodies.get(body_name)
        if body_info and body_info.joint:
            primary_bone = body_info.joint.name
        
        # Get all bones that could influence this mesh
        influence_bones = self._get_influence_bones(body_name)
        
        print(f"  Skinning {mesh_name} (body: {body_name})")
        print(f"    Primary bone: {primary_bone}")
        print(f"    Influence bones: {influence_bones}")
        
        for i, vertex in enumerate(vertices):
            weights = self._compute_vertex_weights(vertex, influence_bones)
            vertex_weights.append(weights)
        
        skinning = MeshSkinning(
            mesh_name=mesh_name,
            body_name=body_name,
            vertex_weights=vertex_weights
        )
        
        self.mesh_skinning[f"{body_name}_{mesh_name}"] = skinning
        return skinning
    
    def _get_influence_bones(self, body_name: str, max_distance: float = 0.3) -> List[str]:
        """
        Get bones that could influence a body's mesh.
        Includes the body's own bone and nearby bones in the hierarchy.
        """
        influence_bones = []
        
        # Always include the body's own bone if it has one
        body_info = self.parser.bodies.get(body_name)
        if body_info and body_info.joint and body_info.joint.name in self.bones:
            influence_bones.append(body_info.joint.name)
        
        # Include parent bones (up the chain)
        current_body = body_name
        for _ in range(3):  # Max 3 levels up
            if current_body in self.parser.bodies:
                parent_name = self.parser.bodies[current_body].parent
                if parent_name and parent_name in self.parser.bodies:
                    parent_body = self.parser.bodies[parent_name]
                    if parent_body.joint and parent_body.joint.name in self.bones:
                        influence_bones.append(parent_body.joint.name)
                    current_body = parent_name
                else:
                    break
            else:
                break
        
        # Include child bones (down the chain)
        self._add_child_bones(body_name, influence_bones, max_depth=2)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_bones = []
        for bone in influence_bones:
            if bone not in seen:
                seen.add(bone)
                unique_bones.append(bone)
        
        return unique_bones
    
    def _add_child_bones(self, body_name: str, influence_bones: List[str], max_depth: int):
        """Add child bones to the influence list."""
        if max_depth <= 0:
            return
        
        if body_name not in self.parser.bodies:
            return
        
        body_info = self.parser.bodies[body_name]
        for child_name in body_info.children:
            if child_name in self.parser.bodies:
                child_body = self.parser.bodies[child_name]
                if child_body.joint and child_body.joint.name in self.bones:
                    influence_bones.append(child_body.joint.name)
                self._add_child_bones(child_name, influence_bones, max_depth - 1)
    
    def _compute_vertex_weights(self, vertex: np.ndarray, influence_bones: List[str]) -> List[Tuple[str, float]]:
        """
        Compute weights for a vertex based on distance to bones.
        Uses exponential falloff and normalizes weights.
        """
        if not influence_bones:
            return []
        
        bone_distances = []
        for bone_name in influence_bones:
            if bone_name in self.bones:
                bone = self.bones[bone_name]
                # Distance from vertex to bone (line segment from head to tail)
                distance = self._point_to_line_distance(vertex, bone.head, bone.tail)
                bone_distances.append((bone_name, distance))
        
        if not bone_distances:
            return []
        
        # Convert distances to weights using exponential falloff
        weights = []
        total_weight = 0.0
        
        for bone_name, distance in bone_distances:
            # Exponential falloff: closer bones have much higher influence
            weight = np.exp(-distance * 20.0)  # Adjust multiplier for falloff rate
            weights.append((bone_name, weight))
            total_weight += weight
        
        # Normalize weights to sum to 1.0
        if total_weight > 0:
            normalized_weights = [(bone_name, weight / total_weight) for bone_name, weight in weights]
            # Filter out very small weights (< 1%)
            significant_weights = [(bone_name, weight) for bone_name, weight in normalized_weights if weight > 0.01]
            
            # Renormalize after filtering
            total_significant = sum(weight for _, weight in significant_weights)
            if total_significant > 0:
                final_weights = [(bone_name, weight / total_significant) for bone_name, weight in significant_weights]
                return final_weights
        
        # Fallback: assign to closest bone only
        if bone_distances:
            closest_bone = min(bone_distances, key=lambda x: x[1])[0]
            return [(closest_bone, 1.0)]
        
        return []
    
    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """
        Calculate the shortest distance from a point to a line segment.
        """
        line_vec = line_end - line_start
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq < 1e-8:  # Very short line, treat as point
            return np.linalg.norm(point - line_start)
        
        # Project point onto line
        t = max(0, min(1, np.dot(point - line_start, line_vec) / line_len_sq))
        projection = line_start + t * line_vec
        
        return np.linalg.norm(point - projection)
    
    def create_skinned_scene(self, mesh_transforms: Dict[str, List], simplified_meshes: Dict[str, trimesh.Trimesh]) -> trimesh.Scene:
        """
        Create a scene with skinned meshes.
        This prepares the data structures for GLB export with proper rigging.
        """
        scene = trimesh.Scene()
        
        print(f"\n=== Creating Skinned Scene ===")
        
        for mesh_name, transforms in mesh_transforms.items():
            if mesh_name not in simplified_meshes:
                continue
            
            print(f"\n--- Skinning mesh: {mesh_name} ---")
            base_mesh = simplified_meshes[mesh_name]
            
            for i, (body_name, position, rotation_matrix, material) in enumerate(transforms):
                print(f"  Instance {i}: body {body_name}")
                
                # Apply transformation to mesh
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                transform_matrix[:3, 3] = position
                
                transformed_mesh = base_mesh.copy()
                transformed_mesh.apply_transform(transform_matrix)
                
                # Assign vertex weights
                skinning = self.assign_vertex_weights(transformed_mesh, body_name, mesh_name)
                
                # Store skinning data in mesh metadata
                instance_name = f"{body_name}_{mesh_name}"
                if len(transforms) > 1:
                    instance_name += f"_instance_{i}"
                
                # Add skinning metadata to the mesh
                if not hasattr(transformed_mesh, 'metadata'):
                    transformed_mesh.metadata = {}
                
                transformed_mesh.metadata['skinning'] = {
                    'vertex_weights': skinning.vertex_weights,
                    'body_name': body_name,
                    'mesh_name': mesh_name
                }
                
                scene.add_geometry(transformed_mesh, node_name=instance_name)
                
                # Print weight statistics
                self._print_weight_stats(skinning)
        
        return scene
    
    def _print_weight_stats(self, skinning: MeshSkinning):
        """Print statistics about vertex weights for debugging."""
        if not skinning.vertex_weights:
            return
        
        bone_usage = {}
        total_vertices = len(skinning.vertex_weights)
        
        for vertex_weights in skinning.vertex_weights:
            for bone_name, weight in vertex_weights:
                if bone_name not in bone_usage:
                    bone_usage[bone_name] = {'count': 0, 'total_weight': 0.0}
                bone_usage[bone_name]['count'] += 1
                bone_usage[bone_name]['total_weight'] += weight
        
        print(f"    Vertex weight statistics:")
        print(f"      Total vertices: {total_vertices}")
        for bone_name, usage in sorted(bone_usage.items(), key=lambda x: x[1]['total_weight'], reverse=True):
            avg_weight = usage['total_weight'] / usage['count'] if usage['count'] > 0 else 0
            print(f"      {bone_name}: {usage['count']} vertices (avg weight: {avg_weight:.3f})")


def create_test_poses(bones: Dict[str, BoneInfo]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create test poses by modifying joint angles.
    Returns a dictionary of pose dictionaries with bone transformations.
    """
    poses = {}
    
    # Pose 1: T-pose (arms out to sides)
    t_pose = {}
    # Rotate shoulder roll joints to move arms horizontally
    if 'left_shoulder_roll_joint' in bones:
        t_pose['left_shoulder_roll_joint'] = np.array([0, 0, np.pi/2])  # 90 degrees
    if 'right_shoulder_roll_joint' in bones:
        t_pose['right_shoulder_roll_joint'] = np.array([0, 0, -np.pi/2])  # -90 degrees
    poses['t_pose'] = t_pose
    
    # Pose 2: Arms forward
    arms_forward = {}
    if 'left_shoulder_pitch_joint' in bones:
        arms_forward['left_shoulder_pitch_joint'] = np.array([np.pi/2, 0, 0])  # 90 degrees forward
    if 'right_shoulder_pitch_joint' in bones:
        arms_forward['right_shoulder_pitch_joint'] = np.array([np.pi/2, 0, 0])  # 90 degrees forward
    poses['arms_forward'] = arms_forward
    
    # Pose 3: Leg lifted
    leg_lift = {}
    if 'left_hip_pitch_joint' in bones:
        leg_lift['left_hip_pitch_joint'] = np.array([np.pi/4, 0, 0])  # 45 degrees up
    if 'left_knee_joint' in bones:
        leg_lift['left_knee_joint'] = np.array([np.pi/3, 0, 0])  # 60 degrees bend
    poses['leg_lift'] = leg_lift
    
    # Pose 4: Waving (elbow bent)
    wave = {}
    if 'right_shoulder_roll_joint' in bones:
        wave['right_shoulder_roll_joint'] = np.array([0, 0, -np.pi/3])  # Arm out
    if 'right_elbow_joint' in bones:
        wave['right_elbow_joint'] = np.array([np.pi/2, 0, 0])  # Elbow bent
    poses['wave'] = wave
    
    return poses


def main():
    """Test the skinning system."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python skinning_utils.py <mjcf_file>")
        return
    
    mjcf_file = sys.argv[1]
    print(f"Testing skinning with {mjcf_file}")
    
    # Parse MJCF and build armature
    parser = MJCFParser(mjcf_file)
    armature_builder = ArmatureBuilder(parser)
    
    # Create skinned mesh builder
    skinned_builder = SkinnedMeshBuilder(parser, armature_builder)
    
    print("Skinning system initialized successfully!")


if __name__ == "__main__":
    main()
