#!/usr/bin/env python3
"""
GLB Armature Exporter
Exports MJCF robot models to GLB format with rigged armatures for Blender.
This creates a proper skeletal hierarchy with skinning that can be used for animation.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import struct
import json
import base64
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation

from mjcf_parser import MJCFParser
from armature_utils import ArmatureBuilder, BoneInfo
from skinning_utils import SkinnedMeshBuilder


@dataclass
class GLTFNode:
    """Represents a node in the glTF scene graph."""
    name: str
    translation: Optional[List[float]] = None
    rotation: Optional[List[float]] = None  # Quaternion [x, y, z, w]
    scale: Optional[List[float]] = None
    children: List[int] = field(default_factory=list)
    mesh: Optional[int] = None
    skin: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to glTF JSON representation."""
        node_dict = {"name": self.name}
        if self.translation:
            node_dict["translation"] = self.translation
        if self.rotation:
            node_dict["rotation"] = self.rotation
        if self.scale:
            node_dict["scale"] = self.scale
        if self.children:
            node_dict["children"] = self.children
        if self.mesh is not None:
            node_dict["mesh"] = self.mesh
        if self.skin is not None:
            node_dict["skin"] = self.skin
        return node_dict


@dataclass 
class GLTFSkin:
    """Represents skinning information in glTF."""
    name: str
    joints: List[int]  # Node indices for joints
    inverse_bind_matrices: int  # Accessor index
    skeleton: Optional[int] = None  # Root joint node index
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to glTF JSON representation."""
        skin_dict = {
            "name": self.name,
            "joints": self.joints,
            "inverseBindMatrices": self.inverse_bind_matrices
        }
        if self.skeleton is not None:
            skin_dict["skeleton"] = self.skeleton
        return skin_dict


class GLBArmatureExporter:
    """Exports MJCF models to GLB format with rigged armatures."""
    
    def __init__(self, parser: MJCFParser):
        self.parser = parser
        self.armature_builder = ArmatureBuilder(parser)
        self.skinning_builder = SkinnedMeshBuilder(parser, self.armature_builder)
        
        # GLB components
        self.nodes: List[GLTFNode] = []
        self.meshes: List[Dict[str, Any]] = []
        self.skins: List[GLTFSkin] = []
        self.accessors: List[Dict[str, Any]] = []
        self.buffer_views: List[Dict[str, Any]] = []
        self.buffers: List[bytes] = []
        
        # Mappings
        self.node_indices: Dict[str, int] = {}
        self.joint_node_indices: Dict[str, int] = {}
        
    def export_to_glb(self, output_path: str, include_mesh_visualization: bool = True) -> None:
        """
        Export the model to GLB format with armature.
        
        Args:
            output_path: Path to save the GLB file
            include_mesh_visualization: Whether to include mesh geometry
        """
        print("Building armature from MJCF...")
        bones = self.armature_builder.compute_bone_positions()
        
        # Create armature nodes
        print("Creating armature nodes...")
        self._create_armature_nodes(bones)
        
        if include_mesh_visualization:
            print("Processing meshes...")
            self._process_meshes()
        
        # Create GLB structure
        print("Creating GLB structure...")
        gltf_data = self._create_gltf_structure()
        
        # Write GLB file
        print(f"Writing GLB file to {output_path}...")
        self._write_glb(gltf_data, output_path)
        
        print(f"✅ GLB with armature exported successfully!")
        
    def _create_armature_nodes(self, bones: Dict[str, BoneInfo]) -> None:
        """Create glTF nodes for the armature bones."""
        # First, create nodes for all bones
        for bone_name, bone_info in bones.items():
            # Calculate bone position relative to parent
            if bone_info.parent and bone_info.parent in bones:
                parent_bone = bones[bone_info.parent]
                parent_pos = parent_bone.head
                relative_pos = bone_info.head - parent_pos
            else:
                # Root bone - use world position
                relative_pos = bone_info.head
                
            # Create node for this joint
            node = GLTFNode(
                name=f"joint_{bone_name}",
                translation=relative_pos.tolist()
            )
            
            node_index = len(self.nodes)
            self.nodes.append(node)
            self.joint_node_indices[bone_name] = node_index
            
        # Second pass: set up parent-child relationships
        for bone_name, bone_info in bones.items():
            if bone_info.parent and bone_info.parent in self.joint_node_indices:
                parent_idx = self.joint_node_indices[bone_info.parent]
                child_idx = self.joint_node_indices[bone_name]
                self.nodes[parent_idx].children.append(child_idx)
                
        # Create a root node for the armature
        root_node = GLTFNode(name="Armature")
        root_node_idx = len(self.nodes)
        self.nodes.append(root_node)
        
        # Add root bones as children of armature node
        for bone_name, bone_info in bones.items():
            if bone_info.parent is None:  # Root bone
                root_node.children.append(self.joint_node_indices[bone_name])
                
    def _process_meshes(self) -> None:
        """Process meshes and create skinned versions."""
        mesh_transforms = self.parser.get_mesh_transforms()
        processed_meshes = {}
        
        for mesh_name, transforms in mesh_transforms.items():
            mesh_file_path = self.parser.get_mesh_file_path(mesh_name)
            if not mesh_file_path or not mesh_file_path.exists():
                continue
                
            # Load base mesh
            try:
                base_mesh = trimesh.load(mesh_file_path)
                if isinstance(base_mesh, trimesh.Scene):
                    base_mesh = base_mesh.dump(concatenate=True)
                    
                # Simplify if needed
                if len(base_mesh.faces) > 5000:
                    base_mesh = base_mesh.simplify_quadric_decimation(face_count=5000)
                    
                processed_meshes[mesh_name] = base_mesh
            except Exception as e:
                print(f"Error loading {mesh_name}: {e}")
                continue
                
            # Process each instance
            for i, (body_name, position, rotation_matrix, material) in enumerate(transforms):
                # Create mesh node
                mesh_node = GLTFNode(
                    name=f"{body_name}_mesh",
                    translation=position.tolist()
                )
                
                # Apply rotation if not identity
                if not np.allclose(rotation_matrix, np.eye(3)):
                    rot = Rotation.from_matrix(rotation_matrix)
                    quat = rot.as_quat()  # [x, y, z, w]
                    mesh_node.rotation = quat.tolist()
                
                # TODO: Add mesh data and skinning
                mesh_node_idx = len(self.nodes)
                self.nodes.append(mesh_node)
                
    def _create_gltf_structure(self) -> Dict[str, Any]:
        """Create the glTF JSON structure."""
        # Basic glTF structure
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "MJCF to GLB Armature Exporter"
            },
            "nodes": [node.to_dict() for node in self.nodes],
            "scenes": [{
                "name": "Scene",
                "nodes": [len(self.nodes) - 1]  # Root armature node
            }],
            "scene": 0
        }
        
        # Add other components if present
        if self.meshes:
            gltf["meshes"] = self.meshes
        if self.skins:
            gltf["skins"] = [skin.to_dict() for skin in self.skins]
        if self.accessors:
            gltf["accessors"] = self.accessors
        if self.buffer_views:
            gltf["bufferViews"] = self.buffer_views
        if self.buffers:
            gltf["buffers"] = [{"byteLength": len(self.buffers[0])}]
            
        return gltf
        
    def _write_glb(self, gltf_data: Dict[str, Any], output_path: str) -> None:
        """Write GLB file with JSON and binary data."""
        # Convert glTF to JSON
        json_data = json.dumps(gltf_data, separators=(',', ':')).encode('utf-8')
        
        # Pad JSON to 4-byte boundary
        json_padding = (4 - len(json_data) % 4) % 4
        json_data += b' ' * json_padding
        
        # Prepare binary buffer (empty for now - just armature)
        binary_data = self.buffers[0] if self.buffers else b''
        binary_padding = (4 - len(binary_data) % 4) % 4
        binary_data += b'\x00' * binary_padding
        
        # GLB Header
        magic = b'glTF'
        version = 2
        length = 12 + 8 + len(json_data) + (8 + len(binary_data) if binary_data else 0)
        
        # Write GLB
        with open(output_path, 'wb') as f:
            # Header
            f.write(magic)
            f.write(struct.pack('<I', version))
            f.write(struct.pack('<I', length))
            
            # JSON chunk
            f.write(struct.pack('<I', len(json_data)))
            f.write(b'JSON')
            f.write(json_data)
            
            # Binary chunk (if present)
            if binary_data:
                f.write(struct.pack('<I', len(binary_data)))
                f.write(b'BIN\x00')
                f.write(binary_data)


def create_rigged_glb(mjcf_path: str = "./g1_description/g1_mjx_alt.xml",
                      output_path: str = "output/robot_rigged.glb") -> None:
    """
    Create a GLB file with rigged armature from MJCF.
    
    Args:
        mjcf_path: Path to MJCF file
        output_path: Output GLB file path
    """
    # Parse MJCF
    parser = MJCFParser(mjcf_path)
    
    # Create exporter and export
    exporter = GLBArmatureExporter(parser)
    exporter.export_to_glb(output_path)
    
    # Also create a visualization of the armature
    armature_builder = ArmatureBuilder(parser)
    armature_scene = armature_builder.create_trimesh_armature()
    if armature_scene:
        armature_vis_path = output_path.replace('.glb', '_armature_vis.glb')
        armature_scene.export(armature_vis_path)
        print(f"Armature visualization saved to: {armature_vis_path}")


if __name__ == "__main__":
    create_rigged_glb()
