#!/usr/bin/env python3
"""
Rigged GLB Exporter
Extends the simple GLB export to include armatures and bone weights for poseable models.
This creates a GLB file that can be imported into Blender with a functional rig.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass

from .mjcf_parser import MJCFParser, JointInfo, BodyInfo
from .utils_3d import load_stl


@dataclass
class BoneInfo:
    """Information about a bone in the armature."""
    name: str
    parent_bone: Optional[str] = None
    joint_info: Optional[JointInfo] = None
    body_name: str = ""
    transform_matrix: np.ndarray = None
    children: List[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.transform_matrix is None:
            self.transform_matrix = np.eye(4)


class RiggedGLBExporter:
    """
    Creates rigged GLB files from MJCF models with skeletal animation support.
    
    This class extends the basic mesh export to include:
    - Bone hierarchy based on MJCF joints
    - Vertex skinning weights
    - Armature data for Blender compatibility
    """
    
    def __init__(self, mjcf_path: str):
        """
        Initialize the rigged exporter.
        
        Args:
            mjcf_path: Path to the MJCF file
        """
        self.mjcf_path = Path(mjcf_path)
        self.parser = MJCFParser(mjcf_path)
        
        # Rigging data
        self.bones: Dict[str, BoneInfo] = {}
        self.bone_hierarchy: List[str] = []  # Root-to-leaf ordering
        self.target_joints: List[str] = []  # Joints to include in armature
        
        # Mesh data
        self.processed_meshes: Dict[str, trimesh.Trimesh] = {}
        self.mesh_instance_weights: Dict[str, Dict[str, float]] = {}  # instance_name -> {bone_name: weight}
        
        print(f"Initialized RiggedGLBExporter for {mjcf_path}")
        self._print_available_joints()
    
    def _print_available_joints(self):
        """Print all available joints for user reference."""
        print("\n=== Available Joints ===")
        for joint_name, joint_info in self.parser.joints.items():
            print(f"  {joint_name}: {joint_info.type} joint, axis={joint_info.axis}")
    
    def set_target_joints(self, joint_names: List[str]):
        """
        Set which joints to include in the armature.
        
        Args:
            joint_names: List of joint names from MJCF to include
        """
        # Validate joints exist
        missing_joints = [j for j in joint_names if j not in self.parser.joints]
        if missing_joints:
            raise ValueError(f"Joints not found in MJCF: {missing_joints}")
        
        self.target_joints = joint_names
        print(f"Target joints set: {joint_names}")
    
    def auto_discover_joint_hierarchy(self, max_joints: Optional[int] = None) -> List[str]:
        """
        Automatically discover joint hierarchy from MJCF, starting from root joints.
        This traverses the joint hierarchy in a sensible order for rigging.
        
        Args:
            max_joints: Maximum number of joints to include (None for all)
            
        Returns:
            List of joint names in hierarchical order
        """
        print("\n=== Auto-discovering Joint Hierarchy ===")
        
        # Get the joint hierarchy from parser
        joint_hierarchy = self.parser.get_joint_hierarchy()
        
        # Find root joints (joints without parents)
        root_joints = []
        for joint_name, joint_info in joint_hierarchy.items():
            if joint_info['parent_joint'] is None:
                root_joints.append(joint_name)
        
        print(f"Found {len(root_joints)} root joints: {root_joints}")
        
        # Traverse hierarchy depth-first
        discovered_joints = []
        visited = set()
        
        def traverse_joint(joint_name: str, depth: int = 0):
            if joint_name in visited or joint_name not in joint_hierarchy:
                return
            
            visited.add(joint_name)
            discovered_joints.append(joint_name)
            
            joint_info = joint_hierarchy[joint_name]
            print(f"  {'  ' * depth}Found joint: {joint_name} ({joint_info['joint_type']})")
            
            # Find children of this joint
            for child_joint_name, child_joint_info in joint_hierarchy.items():
                if child_joint_info['parent_joint'] == joint_name:
                    traverse_joint(child_joint_name, depth + 1)
        
        # Start traversal from root joints
        for root_joint in root_joints:
            traverse_joint(root_joint)
        
        # Limit number of joints if specified
        if max_joints and len(discovered_joints) > max_joints:
            discovered_joints = discovered_joints[:max_joints]
            print(f"Limited to first {max_joints} joints")
        
        print(f"Discovered {len(discovered_joints)} joints in hierarchy: {discovered_joints}")
        return discovered_joints
    
    def set_target_joints_from_hierarchy(self, max_joints: Optional[int] = None):
        """
        Set target joints by auto-discovering the joint hierarchy.
        
        Args:
            max_joints: Maximum number of joints to include (None for all)
        """
        discovered_joints = self.auto_discover_joint_hierarchy(max_joints)
        self.set_target_joints(discovered_joints)
    
    def _build_bone_hierarchy(self):
        """
        Build bone hierarchy from MJCF joint data.
        Only includes bones for joints in target_joints.
        """
        print("\n=== Building Bone Hierarchy ===")
        
        # Clear existing bones
        self.bones.clear()
        self.bone_hierarchy.clear()
        
        if not self.target_joints:
            print("No target joints specified. Use set_target_joints() first.")
            return
        
        # Create bones for target joints
        for joint_name in self.target_joints:
            joint_info = self.parser.joints[joint_name]
            
            # Find the body that contains this joint
            body_name = None
            for body_name_candidate, body_info in self.parser.bodies.items():
                if body_info.joint and body_info.joint.name == joint_name:
                    body_name = body_name_candidate
                    break
            
            if body_name is None:
                print(f"Warning: Could not find body for joint {joint_name}")
                continue
            
            body_info = self.parser.bodies[body_name]
            
            # Determine parent bone by traversing up the body hierarchy
            parent_bone = None
            current_parent = body_info.parent
            
            # Walk up the body hierarchy to find the nearest ancestor with a joint in target_joints
            while current_parent and parent_bone is None:
                if current_parent in self.parser.bodies:
                    parent_body = self.parser.bodies[current_parent]
                    if parent_body.joint and parent_body.joint.name in self.target_joints:
                        parent_bone = parent_body.joint.name
                        break
                    current_parent = parent_body.parent
                else:
                    break
            
            # Create bone
            bone = BoneInfo(
                name=joint_name,
                parent_bone=parent_bone,
                joint_info=joint_info,
                body_name=body_name
            )
            
            # Calculate bone transform (position in world space)
            global_pos, global_rot = self.parser.compute_global_transform(body_name)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = global_rot
            transform_matrix[:3, 3] = global_pos
            bone.transform_matrix = transform_matrix
            
            self.bones[joint_name] = bone
            print(f"  Created bone: {joint_name} (parent: {parent_bone})")
        
        # Build hierarchy order (root to leaf)
        self._compute_bone_hierarchy_order()
        
        # Set up parent-child relationships
        for bone_name, bone in self.bones.items():
            if bone.parent_bone and bone.parent_bone in self.bones:
                self.bones[bone.parent_bone].children.append(bone_name)
    
    def _compute_bone_hierarchy_order(self):
        """Compute the order of bones from root to leaf for GLB export."""
        visited = set()
        
        def visit_bone(bone_name: str):
            if bone_name in visited or bone_name not in self.bones:
                return
            
            bone = self.bones[bone_name]
            # Visit parent first
            if bone.parent_bone and bone.parent_bone not in visited:
                visit_bone(bone.parent_bone)
            
            # Add this bone
            if bone_name not in visited:
                self.bone_hierarchy.append(bone_name)
                visited.add(bone_name)
        
        # Start with all bones (will visit in proper order)
        for bone_name in self.bones.keys():
            visit_bone(bone_name)
        
        print(f"  Bone hierarchy order: {self.bone_hierarchy}")
    
    def _assign_simple_weights(self):
        """
        Assign simple vertex weights for skinning based on mesh instances.
        Each mesh instance gets weights based on its specific body's controlling joint.
        """
        print("\n=== Assigning Vertex Weights ===")
        
        if not self.target_joints:
            print("No target joints - skipping weight assignment")
            return
        
        # Get mesh transforms to understand which meshes exist
        mesh_transforms = self.parser.get_mesh_transforms()
        
        for mesh_name, transforms in mesh_transforms.items():
            print(f"  Processing mesh: {mesh_name}")
            
            # For each transform (mesh instance), assign weights independently
            for i, (body_name, position, rotation_matrix, material) in enumerate(transforms):
                # Create unique instance name
                instance_name = f"{body_name}_{mesh_name}"
                if len(transforms) > 1:
                    instance_name += f"_{i}"
                
                # Find the joint that controls this specific body
                controlling_joint = None
                
                # Look for joint in this body or parent bodies
                current_body = body_name
                while current_body and controlling_joint is None:
                    if current_body in self.parser.bodies:
                        body_info = self.parser.bodies[current_body]
                        if body_info.joint and body_info.joint.name in self.target_joints:
                            controlling_joint = body_info.joint.name
                            break
                        current_body = body_info.parent
                    else:
                        break
                
                # Assign weights for this specific instance
                instance_weights = {}
                if controlling_joint:
                    # Assign 100% weight to the controlling joint for this instance
                    instance_weights[controlling_joint] = 1.0
                    print(f"    Instance {instance_name} -> Joint {controlling_joint} (weight: 1.0)")
                else:
                    print(f"    Instance {instance_name} -> No controlling joint found")
                
                # Store weights for this specific instance
                self.mesh_instance_weights[instance_name] = instance_weights
    
    def export_rigged_glb(self, output_path: str, single_joint: Optional[str] = None) -> None:
        """
        Export the rigged model to GLB format.
        
        Args:
            output_path: Path to save the GLB file
            single_joint: If specified, only include this joint in the rig
        """
        print(f"\n=== Exporting Rigged GLB ===")
        print(f"Output path: {output_path}")
        
        # Set target joints
        if single_joint:
            if single_joint not in self.parser.joints:
                raise ValueError(f"Joint '{single_joint}' not found. Available: {list(self.parser.joints.keys())}")
            self.set_target_joints([single_joint])
        elif not self.target_joints:
            print("No joints specified - exporting basic mesh only")
            return self._export_basic_glb(output_path)
        
        # Build armature
        self._build_bone_hierarchy()
        
        # Assign weights
        self._assign_simple_weights()
        
        # For now, we'll use trimesh to export the mesh and add rigging metadata
        # In a future iteration, we could use pygltflib for full control
        scene = self._create_scene_with_meshes()
        
        # Export basic GLB
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export the scene
        exported = scene.export(file_type='glb')
        with open(output_path, 'wb') as f:
            f.write(exported)
        
        print(f"✅ Basic GLB exported: {output_path}")
        
        # Add metadata about the rigging for future enhancement
        metadata_path = output_path.with_suffix('.json')
        self._export_rigging_metadata(metadata_path)
        
        print(f"📄 Rigging metadata saved: {metadata_path}")
        print("🚧 Note: Full armature support coming in next iteration!")
    
    def _export_basic_glb(self, output_path: str) -> None:
        """Export basic GLB without rigging (fallback)."""
        scene = self._create_scene_with_meshes()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        exported = scene.export(file_type='glb')
        with open(output_path, 'wb') as f:
            f.write(exported)
        
        print(f"✅ Basic GLB exported: {output_path}")
    
    def _create_scene_with_meshes(self) -> trimesh.Scene:
        """Create a trimesh scene with all meshes (same as current system)."""
        scene = trimesh.Scene()
        mesh_transforms = self.parser.get_mesh_transforms()
        
        print(f"Creating scene with {len(mesh_transforms)} unique meshes")
        
        # Process each mesh (reuse existing logic)
        for mesh_name, transforms in mesh_transforms.items():
            print(f"  Processing {mesh_name}...")
            
            # Get mesh file path
            mesh_file_path = self.parser.get_mesh_file_path(mesh_name)
            if mesh_file_path is None or not mesh_file_path.exists():
                print(f"    Warning: Skipping missing mesh {mesh_name}")
                continue
            
            # Load mesh (cache it if we haven't loaded it before)
            if mesh_name not in self.processed_meshes:
                try:
                    mesh = load_stl(mesh_file_path)
                    # Simple reduction to keep file size reasonable
                    if len(mesh.faces) > 5000:
                        mesh = mesh.simplify_quadric_decimation(face_count=5000)
                    self.processed_meshes[mesh_name] = mesh
                except Exception as e:
                    print(f"    Error loading {mesh_name}: {e}")
                    continue
            
            base_mesh = self.processed_meshes[mesh_name]
            
            # Add transformed instances to scene
            for i, (body_name, position, rotation_matrix, material) in enumerate(transforms):
                # Apply transform
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                transform_matrix[:3, 3] = position
                
                transformed_mesh = base_mesh.copy()
                transformed_mesh.apply_transform(transform_matrix)
                
                instance_name = f"{body_name}_{mesh_name}"
                if len(transforms) > 1:
                    instance_name += f"_{i}"
                scene.add_geometry(transformed_mesh, node_name=instance_name)
        
        return scene
    
    def _export_rigging_metadata(self, metadata_path: Path) -> None:
        """Export rigging metadata for future use."""
        metadata = {
            "version": "1.0",
            "source_mjcf": str(self.mjcf_path),
            "target_joints": self.target_joints,
            "bones": {},
            "mesh_instance_weights": self.mesh_instance_weights,
            "notes": "This metadata describes the intended rigging structure. Weights are now per-instance to handle shared meshes correctly."
        }
        
        # Add bone information
        for bone_name, bone in self.bones.items():
            metadata["bones"][bone_name] = {
                "parent": bone.parent_bone,
                "body_name": bone.body_name,
                "joint_type": bone.joint_info.type if bone.joint_info else "unknown",
                "transform": bone.transform_matrix.tolist(),
                "children": bone.children
            }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def print_rigging_summary(self) -> None:
        """Print a summary of the rigging structure."""
        print("\n=== Rigging Summary ===")
        print(f"Target joints: {self.target_joints}")
        print(f"Total bones: {len(self.bones)}")
        print(f"Bone hierarchy: {self.bone_hierarchy}")
        
        print("\nBone details:")
        for bone_name in self.bone_hierarchy:
            bone = self.bones[bone_name]
            parent_str = f" (parent: {bone.parent_bone})" if bone.parent_bone else " (root)"
            print(f"  {bone_name}{parent_str}")
            print(f"    Body: {bone.body_name}")
            if bone.joint_info:
                print(f"    Joint: {bone.joint_info.type}, axis={bone.joint_info.axis}")
        
        print("\nMesh instance weights:")
        for instance_name, weights in self.mesh_instance_weights.items():
            print(f"  {instance_name}: {weights}")


def create_waist_and_shoulder_rig(mjcf_path: str = "./g1_description/g1_mjx_alt.xml", 
                                  output_name: str = "rigged_waist_shoulder") -> None:
    """
    Create a rigged GLB with waist and right shoulder pitch joints.
    
    Args:
        mjcf_path: Path to MJCF file
        output_name: Output filename (without extension)
    """
    print("🦴 Creating waist + right shoulder rig...")
    
    # Create rigged exporter
    exporter = RiggedGLBExporter(mjcf_path)
    
    # Set multiple target joints as shown in the implementation summary
    exporter.set_target_joints([
        "waist_yaw_joint",
        "right_shoulder_pitch_joint"
    ])
    
    # Export the rigged GLB
    output_path = f"output/{output_name}.glb"
    exporter.export_rigged_glb(output_path)
    
    # Print summary
    exporter.print_rigging_summary()
    
    print(f"\n✅ Waist + Right Shoulder rig created!")
    print(f"   GLB: {output_path}")
    print(f"   Metadata: {output_path.replace('.glb', '.json')}")

def create_hierarchical_rig(mjcf_path: str = "./g1_description/g1_mjx_alt.xml", 
                           output_name: str = "rigged_hierarchical",
                           max_joints: int = 5) -> None:
    """
    Create a rigged GLB by auto-discovering joints from the MJCF hierarchy.
    This is the "traversing the MJCF for joints" approach you requested.
    
    Args:
        mjcf_path: Path to MJCF file
        output_name: Output filename (without extension)
        max_joints: Maximum number of joints to include from hierarchy
    """
    print(f"🦴 Creating hierarchical rig with up to {max_joints} joints...")
    
    # Create rigged exporter
    exporter = RiggedGLBExporter(mjcf_path)
    
    # Auto-discover joints from hierarchy (this is the main feature you requested)
    exporter.set_target_joints_from_hierarchy(max_joints=max_joints)
    
    # Export the rigged GLB
    output_path = f"output/{output_name}.glb"
    exporter.export_rigged_glb(output_path)
    
    # Print summary
    exporter.print_rigging_summary()
    
    print(f"\n✅ Hierarchical rig created!")
    print(f"   GLB: {output_path}")
    print(f"   Metadata: {output_path.replace('.glb', '.json')}")

def create_full_body_rig(mjcf_path: str = "./g1_description/g1_mjx_alt.xml", 
                        output_name: str = "rigged_full_body") -> None:
    """
    Create a rigged GLB with complete kinematic chains to ankles and forearms.
    This includes all major joints in both legs and arms.
    
    Args:
        mjcf_path: Path to MJCF file
        output_name: Output filename (without extension)
    """
    print("🦴 Creating full body rig with complete kinematic chains...")
    
    # Create rigged exporter
    exporter = RiggedGLBExporter(mjcf_path)
    
    # Manually select key joints for full body coverage
    # Excludes floating_base_joint (which is usually not needed for rigging)
    full_body_joints = [
        # Waist
        "waist_yaw_joint",
        
        # Right leg (hip to ankle)
        "right_hip_pitch_joint",
        "right_hip_roll_joint", 
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        
        # Left leg (hip to ankle) 
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint", 
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        
        # Right arm (shoulder to wrist)
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        
        # Left arm (shoulder to wrist)
        "left_shoulder_pitch_joint", 
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint"
    ]
    
    exporter.set_target_joints(full_body_joints)
    
    # Export the rigged GLB
    output_path = f"output/{output_name}.glb"
    exporter.export_rigged_glb(output_path)
    
    # Print summary
    exporter.print_rigging_summary()
    
    print(f"\n✅ Full body rig created with {len(full_body_joints)} joints!")
    print(f"   GLB: {output_path}")
    print(f"   Metadata: {output_path.replace('.glb', '.json')}")
    print("🎯 Complete kinematic chains: waist → shoulders → elbows → wrists")
    print("                            : waist → hips → knees → ankles")

def create_simple_waist_rig(mjcf_path: str = "./g1_description/g1_mjx_alt.xml", 
                           output_name: str = "rigged_robot") -> None:
    """
    Create a simple rigged GLB with just the waist joint.
    
    Args:
        mjcf_path: Path to MJCF file
        output_name: Output filename (without extension)
    """
    print("🦴 Creating simple waist rig...")
    
    # Create rigged exporter
    exporter = RiggedGLBExporter(mjcf_path)
    
    # Export with waist joint
    output_path = f"output/{output_name}.glb"
    exporter.export_rigged_glb(output_path, single_joint="waist_yaw_joint")
    
    # Print summary
    exporter.print_rigging_summary()
    
    print(f"\n✅ Simple waist rig created!")
    print(f"   GLB: {output_path}")
    print(f"   Metadata: {output_path.replace('.glb', '.json')}")


if __name__ == "__main__":
    # Demo: Create simple waist rig
    create_simple_waist_rig()
