#!/usr/bin/env python3
"""
GLTF Armature Builder
Creates GLB files with proper armature/skeleton support using pygltflib.
This enables full rigging with bones, joints, and vertex weights.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import struct
from dataclasses import dataclass
import base64

try:
    from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Accessor, BufferView, Buffer
    from pygltflib import Material, Skin, Animation, AnimationChannel, AnimationSampler
    from pygltflib import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
    from pygltflib import FLOAT, UNSIGNED_SHORT, UNSIGNED_INT
    from pygltflib import JOINTS_0, WEIGHTS_0, POSITION, NORMAL
    PYGLTFLIB_AVAILABLE = True
except ImportError:
    PYGLTFLIB_AVAILABLE = False
    print("Warning: pygltflib not available. Install with: uv pip install pygltflib")

from .mjcf_parser import MJCFParser, JointInfo
from dataclasses import dataclass


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


class GLTFArmatureBuilder:
    """
    Builds proper GLTF files with armature support using pygltflib.
    This creates GLB files that can be imported into Blender with working rigs.
    """
    
    def __init__(self, mjcf_parser: MJCFParser, bones: Dict[str, BoneInfo], 
                 bone_hierarchy: List[str], mesh_instance_weights: Dict[str, Dict[str, float]]):
        """Initialize with bone structure data."""
        if not PYGLTFLIB_AVAILABLE:
            raise ImportError("pygltflib is required. Install with: uv pip install pygltflib")
        
        self.parser = mjcf_parser
        self.bones = bones
        self.bone_hierarchy = bone_hierarchy
        self.mesh_instance_weights = mesh_instance_weights
        self.gltf = GLTF2()
        
        # Buffers and accessors
        self.buffer_data = bytearray()
        self.current_buffer_offset = 0
        
        # Mappings
        self.bone_to_joint_index: Dict[str, int] = {}
        self.mesh_to_primitive: Dict[str, int] = {}
        
        print("GLTFArmatureBuilder initialized")
    
    def _add_buffer_data(self, data: bytes) -> Tuple[int, int]:
        """Add data to buffer and return (offset, length)."""
        offset = len(self.buffer_data)
        self.buffer_data.extend(data)
        
        # Align to 4-byte boundary
        while len(self.buffer_data) % 4 != 0:
            self.buffer_data.extend(b'\x00')
        
        return offset, len(data)
    
    def _create_accessor(self, buffer_view_index: int, component_type: int, 
                        count: int, type_: str, max_vals: List[float] = None, 
                        min_vals: List[float] = None) -> int:
        """Create an accessor and return its index."""
        accessor = Accessor(
            bufferView=buffer_view_index,
            componentType=component_type,
            count=count,
            type=type_,
            max=max_vals,
            min=min_vals
        )
        self.gltf.accessors.append(accessor)
        return len(self.gltf.accessors) - 1
    
    def _create_buffer_view(self, buffer_index: int, byte_offset: int, 
                           byte_length: int, target: int = None) -> int:
        """Create a buffer view and return its index."""
        buffer_view = BufferView(
            buffer=buffer_index,
            byteOffset=byte_offset,
            byteLength=byte_length,
            target=target
        )
        self.gltf.bufferViews.append(buffer_view)
        return len(self.gltf.bufferViews) - 1
    
    def _add_mesh_to_gltf(self, mesh_name: str, mesh: trimesh.Trimesh, 
                         transform: np.ndarray) -> int:
        """Add a mesh to the GLTF and return the mesh index."""
        # Apply coordinate system correction for Blender (MJCF -> Blender)
        blender_correction = np.array([
            [1,  0,  0,  0],  # X stays X
            [0,  0,  1,  0],  # Y becomes Z 
            [0, -1,  0,  0],  # Z becomes -Y
            [0,  0,  0,  1]   # Translation unchanged
        ])
        corrected_transform = blender_correction @ transform
        
        # Apply transform to mesh
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(corrected_transform)
        
        # Get mesh data
        vertices = transformed_mesh.vertices.astype(np.float32)
        faces = transformed_mesh.faces.astype(np.uint32)
        normals = transformed_mesh.vertex_normals.astype(np.float32)
        
        # Create vertex buffer data
        vertex_data = vertices.tobytes()
        normal_data = normals.tobytes()
        index_data = faces.flatten().tobytes()
        
        # Add to buffer
        vertex_offset, vertex_length = self._add_buffer_data(vertex_data)
        normal_offset, normal_length = self._add_buffer_data(normal_data)
        index_offset, index_length = self._add_buffer_data(index_data)
        
        # Create buffer views
        vertex_buffer_view = self._create_buffer_view(0, vertex_offset, vertex_length, ARRAY_BUFFER)
        normal_buffer_view = self._create_buffer_view(0, normal_offset, normal_length, ARRAY_BUFFER)
        index_buffer_view = self._create_buffer_view(0, index_offset, index_length, ELEMENT_ARRAY_BUFFER)
        
        # Create accessors
        vertex_accessor = self._create_accessor(
            vertex_buffer_view, FLOAT, len(vertices), "VEC3",
            max_vals=vertices.max(axis=0).tolist(),
            min_vals=vertices.min(axis=0).tolist()
        )
        
        normal_accessor = self._create_accessor(
            normal_buffer_view, FLOAT, len(normals), "VEC3"
        )
        
        index_accessor = self._create_accessor(
            index_buffer_view, UNSIGNED_INT, len(faces) * 3, "SCALAR",
            max_vals=[int(faces.max())],
            min_vals=[int(faces.min())]
        )
        
        # Add vertex weights if this mesh instance has bone weights
        joint_accessor = None
        weight_accessor = None
        
        # Generate instance name to look up weights
        # Note: mesh_name here is the original mesh name, we need to match the instance naming
        # This will be set by the calling code that knows the instance name
        instance_weights = getattr(self, '_current_instance_weights', {})
        
        if instance_weights:
            # Create joint and weight data for this mesh instance
            joints, weight_values = self._create_vertex_weights(len(vertices), instance_weights)
            
            # Add to buffer
            joint_data = joints.astype(np.uint16).tobytes()
            weight_data = weight_values.astype(np.float32).tobytes()
            
            joint_offset, joint_length = self._add_buffer_data(joint_data)
            weight_offset, weight_length = self._add_buffer_data(weight_data)
            
            # Create buffer views and accessors
            joint_buffer_view = self._create_buffer_view(0, joint_offset, joint_length, ARRAY_BUFFER)
            weight_buffer_view = self._create_buffer_view(0, weight_offset, weight_length, ARRAY_BUFFER)
            
            joint_accessor = self._create_accessor(joint_buffer_view, UNSIGNED_SHORT, len(vertices), "VEC4")
            weight_accessor = self._create_accessor(weight_buffer_view, FLOAT, len(vertices), "VEC4")
        
        # Create primitive
        attributes = {
            POSITION: vertex_accessor,
            NORMAL: normal_accessor
        }
        
        if joint_accessor is not None and weight_accessor is not None:
            attributes[JOINTS_0] = joint_accessor
            attributes[WEIGHTS_0] = weight_accessor
        
        primitive = Primitive(
            attributes=attributes,
            indices=index_accessor,
            material=0  # We'll create a default material
        )
        
        # Create mesh
        gltf_mesh = Mesh(primitives=[primitive])
        self.gltf.meshes.append(gltf_mesh)
        
        return len(self.gltf.meshes) - 1
    
    def _create_vertex_weights(self, num_vertices: int, 
                              bone_weights: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create vertex joint indices and weights for skinning.
        
        Returns:
            joints: Array of shape (num_vertices, 4) with joint indices
            weights: Array of shape (num_vertices, 4) with weight values
        """
        # Initialize arrays
        joints = np.zeros((num_vertices, 4), dtype=np.uint16)
        weights = np.zeros((num_vertices, 4), dtype=np.float32)
        
        # For this simple implementation, assign the same weights to all vertices
        # In a more sophisticated version, you'd calculate weights based on distance
        # to bones, mesh topology, etc.
        
        bone_indices = []
        weight_values = []
        
        for bone_name, weight in bone_weights.items():
            if bone_name in self.bone_to_joint_index:
                bone_indices.append(self.bone_to_joint_index[bone_name])
                weight_values.append(weight)
        
        # Normalize weights
        if weight_values:
            total_weight = sum(weight_values)
            if total_weight > 0:
                weight_values = [w / total_weight for w in weight_values]
        
        # Assign to all vertices (simple approach)
        for i in range(num_vertices):
            for j, (bone_idx, weight_val) in enumerate(zip(bone_indices, weight_values)):
                if j < 4:  # GLTF supports up to 4 influences per vertex
                    joints[i, j] = bone_idx
                    weights[i, j] = weight_val
        
        return joints, weights
    
    def _create_armature_nodes(self) -> List[int]:
        """Create armature nodes and return list of joint indices."""
        joint_nodes = []
        
        # Create nodes for each bone in hierarchy order
        for bone_name in self.bone_hierarchy:
            bone = self.bones[bone_name]
            
            # Create node for this joint
            node = Node(name=bone_name)
            
            # Apply coordinate system correction for Blender
            blender_correction = np.array([
                [1,  0,  0,  0],  # X stays X
                [0,  0,  1,  0],  # Y becomes Z 
                [0, -1,  0,  0],  # Z becomes -Y
                [0,  0,  0,  1]   # Translation unchanged
            ])
            
            # Calculate bone position relative to parent
            if bone.parent_bone and bone.parent_bone in self.bones:
                # Child bone: position relative to parent
                parent_bone = self.bones[bone.parent_bone]
                
                # Get corrected global transforms (these already include all intermediate bodies)
                corrected_bone_transform = blender_correction @ bone.transform_matrix
                corrected_parent_transform = blender_correction @ parent_bone.transform_matrix
                
                # Calculate relative position: difference in world positions
                # This works because the parser already computed global positions through all intermediate bodies
                bone_world_pos = corrected_bone_transform[:3, 3]
                parent_world_pos = corrected_parent_transform[:3, 3]
                
                # For GLTF bones, we need the local offset from parent to child
                # Since both positions are in the same coordinate system, simple subtraction works
                translation = (bone_world_pos - parent_world_pos).tolist()
            else:
                # Root bone: use world position
                corrected_transform = blender_correction @ bone.transform_matrix
                translation = corrected_transform[:3, 3].tolist()
                
                # Adjust root bone position for better placement
                if bone_name == "waist_yaw_joint":
                    translation[2] = 0.0  # At waist level after coordinate correction
            
            node.translation = translation
            
            # Keep bones simple - no rotation modifications needed
            
            # Set parent relationship
            if bone.parent_bone and bone.parent_bone in self.bone_to_joint_index:
                parent_index = self.bone_to_joint_index[bone.parent_bone]
                parent_node = self.gltf.nodes[parent_index]
                if parent_node.children is None:
                    parent_node.children = []
                parent_node.children.append(len(self.gltf.nodes))
            
            self.gltf.nodes.append(node)
            joint_index = len(self.gltf.nodes) - 1
            joint_nodes.append(joint_index)
            self.bone_to_joint_index[bone_name] = joint_index
        
        return joint_nodes
    
    def _create_skin(self, joint_nodes: List[int]) -> int:
        """Create skin for the armature and return skin index."""
        # Create inverse bind matrices (identity for simplicity)
        num_joints = len(joint_nodes)
        inverse_bind_matrices = np.tile(np.eye(4, dtype=np.float32), (num_joints, 1, 1))
        
        # Add to buffer
        matrix_data = inverse_bind_matrices.tobytes()
        matrix_offset, matrix_length = self._add_buffer_data(matrix_data)
        
        # Create buffer view and accessor
        matrix_buffer_view = self._create_buffer_view(0, matrix_offset, matrix_length)
        matrix_accessor = self._create_accessor(matrix_buffer_view, FLOAT, num_joints, "MAT4")
        
        # Create skin
        skin = Skin(
            joints=joint_nodes,
            inverseBindMatrices=matrix_accessor
        )
        self.gltf.skins.append(skin)
        
        return len(self.gltf.skins) - 1
    
    def build_rigged_gltf(self, output_path: str) -> None:
        """Build the complete rigged GLTF file."""
        print(f"\n=== Building Rigged GLTF ===")
        print(f"Output: {output_path}")
        
        # Initialize GLTF structure
        self.gltf.scenes = [Scene(nodes=[])]
        self.gltf.scene = 0
        
        # Create default material
        material = Material(name="DefaultMaterial")
        self.gltf.materials = [material]
        
        # Create armature nodes
        print("Creating armature...")
        joint_nodes = self._create_armature_nodes()
        
        # Create skin
        print("Creating skin...")
        skin_index = self._create_skin(joint_nodes)
        
        # Add meshes
        print("Adding meshes...")
        mesh_transforms = self.parser.get_mesh_transforms_detailed()
        
        for mesh_name, transforms in mesh_transforms.items():
            mesh_file_path = self.parser.get_mesh_file_path(mesh_name)
            if mesh_file_path is None or not mesh_file_path.exists():
                continue
            
            # Load mesh
            try:
                # Use trimesh directly to avoid import issues
                mesh = trimesh.load(str(mesh_file_path))
                if isinstance(mesh, trimesh.Scene):
                    # If it's a scene, get the first mesh
                    if len(mesh.geometry) == 0:
                        print(f"Warning: No geometry in {mesh_name}")
                        continue
                    mesh = list(mesh.geometry.values())[0]
                
                if not isinstance(mesh, trimesh.Trimesh):
                    print(f"Warning: {mesh_name} is not a valid mesh")
                    continue
                    
                # Simplify if needed
                if len(mesh.faces) > 5000:
                    mesh = mesh.simplify_quadric_decimation(face_count=5000)
                    
            except Exception as e:
                print(f"Error loading {mesh_name}: {e}")
                continue
            
            # Add each transform instance
            for i, (body_name, position, rotation_matrix, material, geom_pos, geom_quat) in enumerate(transforms):
                # Create instance name for weight lookup
                instance_name = f"{body_name}_{mesh_name}"
                if len(transforms) > 1:
                    instance_name += f"_{i}"
                
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                transform_matrix[:3, 3] = position
                
                # For rigged meshes, adjust transform to be relative to bone
                if instance_name in self.mesh_instance_weights and self.mesh_instance_weights[instance_name]:
                    # This mesh instance is controlled by a bone - make transform relative to bone
                    controlling_bones = list(self.mesh_instance_weights[instance_name].keys())
                    if controlling_bones:
                        controlling_bone_name = controlling_bones[0]  # Use first controlling bone
                        if controlling_bone_name in self.bones:
                            bone = self.bones[controlling_bone_name]
                            
                            # Get bone transform (with coordinate correction)
                            blender_correction = np.array([
                                [1,  0,  0,  0],  # X stays X
                                [0,  0,  1,  0],  # Y becomes Z 
                                [0, -1,  0,  0],  # Z becomes -Y
                                [0,  0,  0,  1]   # Translation unchanged
                            ])
                            bone_transform = blender_correction @ bone.transform_matrix
                            
                            # Make mesh transform relative to bone by removing bone's transform
                            try:
                                bone_inverse = np.linalg.inv(bone_transform)
                                transform_matrix = bone_inverse @ blender_correction @ transform_matrix
                            except np.linalg.LinAlgError:
                                # If bone transform is not invertible, use original transform
                                pass
                
                # Set current instance weights for the mesh creation
                if instance_name in self.mesh_instance_weights:
                    self._current_instance_weights = self.mesh_instance_weights[instance_name]
                else:
                    self._current_instance_weights = {}
                
                # Create parent transform node for EVERY mesh (for independent positioning)
                transform_node_name = f"{instance_name}_transform"
                transform_node = Node(name=transform_node_name)
                
                self.gltf.nodes.append(transform_node)
                transform_node_index = len(self.gltf.nodes) - 1
                
                # Add mesh to GLTF with the existing transform (which is already processed correctly)
                mesh_index = self._add_mesh_to_gltf(mesh_name, mesh, transform_matrix)
                
                # Create mesh node as child of transform node
                mesh_node = Node(name=instance_name, mesh=mesh_index)
                
                # Apply geom-specific offsets to the mesh node if present
                if geom_pos:
                    # Apply coordinate correction to geom position
                    blender_correction = np.array([
                        [1,  0,  0,  0],  # X stays X
                        [0,  0,  1,  0],  # Y becomes Z 
                        [0, -1,  0,  0],  # Z becomes -Y
                        [0,  0,  0,  1]   # Translation unchanged
                    ])
                    geom_pos_4d = np.array([*geom_pos, 1.0])
                    corrected_geom_pos = (blender_correction @ geom_pos_4d)[:3]
                    mesh_node.translation = corrected_geom_pos.tolist()
                
                if geom_quat:
                    # Convert MJCF quaternion (w,x,y,z) to GLTF quaternion (x,y,z,w)
                    # Apply coordinate correction
                    from scipy.spatial.transform import Rotation
                    geom_rotation = Rotation.from_quat([geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]])  # MJCF w,x,y,z -> scipy x,y,z,w
                    geom_rot_matrix = geom_rotation.as_matrix()
                    
                    # Apply Blender coordinate correction
                    blender_correction_3x3 = np.array([
                        [1,  0,  0],   # X stays X
                        [0,  0,  1],   # Y becomes Z 
                        [0, -1,  0]    # Z becomes -Y
                    ])
                    corrected_geom_rot = blender_correction_3x3 @ geom_rot_matrix @ blender_correction_3x3.T
                    corrected_rotation = Rotation.from_matrix(corrected_geom_rot)
                    mesh_node.rotation = corrected_rotation.as_quat().tolist()  # Returns x,y,z,w for GLTF
                
                # Add skin to mesh if it has weights
                if instance_name in self.mesh_instance_weights and self.mesh_instance_weights[instance_name]:
                    mesh_node.skin = skin_index
                
                self.gltf.nodes.append(mesh_node)
                mesh_node_index = len(self.gltf.nodes) - 1
                
                # Make mesh a child of the transform node
                transform_node.children = [mesh_node_index]
                
                # Add ONLY transform node to scene (mesh is its child, so doesn't need to be added separately)
                self.gltf.scenes[0].nodes.append(transform_node_index)
        
        # Add armature root to scene
        if joint_nodes:
            # Find root joints (those without parents)
            root_joints = []
            for bone_name in self.bone_hierarchy:
                bone = self.bones[bone_name]
                if bone.parent_bone is None:
                    root_joints.append(self.bone_to_joint_index[bone_name])
            
            for root_joint in root_joints:
                self.gltf.scenes[0].nodes.append(root_joint)
        
        # Create buffer
        buffer = Buffer(byteLength=len(self.buffer_data))
        self.gltf.buffers = [buffer]
        
        # Convert buffer data to base64 for embedded GLB
        buffer_uri = "data:application/octet-stream;base64," + base64.b64encode(self.buffer_data).decode('ascii')
        buffer.uri = buffer_uri
        
        # Save GLTF
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.gltf.save(str(output_path))
        print(f"✅ Rigged GLB saved: {output_path}")
        
        # Print summary
        print(f"📊 Summary:")
        print(f"   Joints: {len(joint_nodes)}")
        print(f"   Meshes: {len(self.gltf.meshes)}")
        print(f"   Nodes: {len(self.gltf.nodes)}")
        print(f"   Buffer size: {len(self.buffer_data)} bytes")


def build_bone_hierarchy(parser: MJCFParser, target_joints: List[str]) -> Tuple[Dict[str, BoneInfo], List[str], Dict[str, Dict[str, float]]]:
    """
    Build bone hierarchy from MJCF joint data.
    
    Args:
        parser: MJCF parser instance
        target_joints: List of joint names to include
        
    Returns:
        Tuple of (bones dict, bone hierarchy list, mesh instance weights)
    """
    bones = {}
    mesh_instance_weights = {}
    
    # Create bones for target joints
    for joint_name in target_joints:
        joint_info = parser.joints[joint_name]
        
        # Find the body that contains this joint
        body_name = None
        for body_name_candidate, body_info in parser.bodies.items():
            if body_info.joint and body_info.joint.name == joint_name:
                body_name = body_name_candidate
                break
        
        if body_name is None:
            print(f"Warning: Could not find body for joint {joint_name}")
            continue
        
        body_info = parser.bodies[body_name]
        
        # Determine parent bone by traversing up the body hierarchy
        parent_bone = None
        current_parent = body_info.parent
        
        # Walk up the body hierarchy to find the nearest ancestor with a joint in target_joints
        while current_parent and parent_bone is None:
            if current_parent in parser.bodies:
                parent_body = parser.bodies[current_parent]
                if parent_body.joint and parent_body.joint.name in target_joints:
                    parent_bone = parent_body.joint.name
                    break
                current_parent = parent_body.parent
            else:
                break
        
        # Calculate bone transform (position in world space)
        global_pos, global_rot = parser.compute_global_transform(body_name)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = global_rot
        transform_matrix[:3, 3] = global_pos
        
        # Create bone
        bone = BoneInfo(
            name=joint_name,
            parent_bone=parent_bone,
            joint_info=joint_info,
            body_name=body_name,
            transform_matrix=transform_matrix
        )
        
        bones[joint_name] = bone
    
    # Build hierarchy order (root to leaf)
    bone_hierarchy = []
    visited = set()
    
    def visit_bone(bone_name: str):
        if bone_name in visited or bone_name not in bones:
            return
        
        bone = bones[bone_name]
        # Visit parent first
        if bone.parent_bone and bone.parent_bone not in visited:
            visit_bone(bone.parent_bone)
        
        # Add this bone
        if bone_name not in visited:
            bone_hierarchy.append(bone_name)
            visited.add(bone_name)
    
    # Start with all bones (will visit in proper order)
    for bone_name in bones.keys():
        visit_bone(bone_name)
    
    # Set up parent-child relationships
    for bone_name, bone in bones.items():
        if bone.parent_bone and bone.parent_bone in bones:
            bones[bone.parent_bone].children.append(bone_name)
    
    # Assign simple vertex weights
    mesh_transforms = parser.get_mesh_transforms()
    
    for mesh_name, transforms in mesh_transforms.items():
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
                if current_body in parser.bodies:
                    body_info = parser.bodies[current_body]
                    if body_info.joint and body_info.joint.name in target_joints:
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
            
            # Store weights for this specific instance
            mesh_instance_weights[instance_name] = instance_weights
    
    return bones, bone_hierarchy, mesh_instance_weights


def create_rigged_full_body_glb(mjcf_path: str = "./g1_description/g1_mjx_alt.xml", 
                               output_name: str = "robot_rigged") -> None:
    """
    Create a fully rigged GLB with complete kinematic chains to ankles and forearms.
    This creates a comprehensive rig suitable for full body animation.
    
    Args:
        mjcf_path: Path to MJCF file
        output_name: Output filename (without extension)
    """
    print("🦴 Creating full body rigged GLB with complete kinematic chains...")
    
    # Parse MJCF
    parser = MJCFParser(mjcf_path)
    
    # Set full body joints
    full_body_joints = [
        "waist_yaw_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", 
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint"
    ]
    
    # Build bone hierarchy
    bones, bone_hierarchy, mesh_instance_weights = build_bone_hierarchy(parser, full_body_joints)
    
    # Create GLTF builder
    builder = GLTFArmatureBuilder(parser, bones, bone_hierarchy, mesh_instance_weights)
    
    # Build and save
    output_path = f"output/{output_name}.glb"
    builder.build_rigged_gltf(output_path)
    
    print(f"\n✅ Full rigged GLB created with {len(full_body_joints)} joints!")
    print(f"   File: {output_path}")
    print(f"   Size: {Path(output_path).stat().st_size / (1024*1024):.1f} MB")
    print("🎯 Complete kinematic chains: waist → shoulders → elbows → wrists")
    print("                            : waist → hips → knees → ankles")
    print("🦴 Armature: Full GLTF armature with proper skinning")
    print("📋 Ready for Blender import and animation!")


if __name__ == "__main__":
    # Create full rigged GLB
    create_rigged_full_body_glb()
