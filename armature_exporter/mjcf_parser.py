#!/usr/bin/env python3
"""
MJCF Parser
This module parses MuJoCo XML files (MJCF) to extract mesh information,
body hierarchy, positions, and orientations for 3D model conversion.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class MeshInfo:
    """Information about a mesh from MJCF asset definition."""
    name: str
    file: str
    scale: Optional[Tuple[float, float, float]] = None


@dataclass
class GeomInfo:
    """Information about a geometry element."""
    mesh_name: Optional[str] = None
    position: Optional[Tuple[float, float, float]] = None
    quaternion: Optional[Tuple[float, float, float, float]] = None
    material: Optional[str] = None
    class_: Optional[str] = None


@dataclass
class JointInfo:
    """Information about a joint in the MJCF model."""
    name: str
    type: str = "hinge"  # hinge, slide, ball, free, etc.
    axis: Optional[Tuple[float, float, float]] = None
    range: Optional[Tuple[float, float]] = None
    damping: Optional[float] = None
    stiffness: Optional[float] = None
    limited: bool = False


@dataclass
class BodyInfo:
    """Information about a body in the MJCF hierarchy."""
    name: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z
    parent: Optional[str] = None
    children: List[str] = None
    geometries: List[GeomInfo] = None
    joint: Optional[JointInfo] = None  # Joint connecting this body to its parent
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.geometries is None:
            self.geometries = []


class MJCFParser:
    """Parser for MuJoCo XML Format (MJCF) files."""
    
    def __init__(self, mjcf_path: Union[str, Path]):
        self.mjcf_path = Path(mjcf_path)
        self.tree = ET.parse(self.mjcf_path)
        self.root = self.tree.getroot()
        
        # Parsed data
        self.meshes: Dict[str, MeshInfo] = {}
        self.bodies: Dict[str, BodyInfo] = {}
        self.materials: Dict[str, Dict] = {}
        self.joints: Dict[str, JointInfo] = {}  # All joints indexed by name
        
        # Parse the MJCF file
        self._parse_assets()
        self._parse_materials()
        self._parse_worldbody()
    
    def _parse_vector3(self, text: str, default: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
        """Parse a 3D vector from text."""
        if not text:
            return default
        try:
            values = [float(x) for x in text.strip().split()]
            if len(values) >= 3:
                return (values[0], values[1], values[2])
            elif len(values) == 1:
                return (values[0], values[0], values[0])
            else:
                return default
        except (ValueError, IndexError):
            return default
    
    def _parse_quaternion(self, text: str, default: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)) -> Tuple[float, float, float, float]:
        """Parse a quaternion from text. MJCF format is w, x, y, z."""
        if not text:
            return default
        try:
            values = [float(x) for x in text.strip().split()]
            if len(values) >= 4:
                return (values[0], values[1], values[2], values[3])  # w, x, y, z
            else:
                return default
        except (ValueError, IndexError):
            return default
    
    def _parse_assets(self):
        """Parse the asset section to extract mesh definitions."""
        asset_elem = self.root.find('asset')
        if asset_elem is not None:
            for mesh_elem in asset_elem.findall('mesh'):
                name = mesh_elem.get('name')
                file = mesh_elem.get('file')
                scale = mesh_elem.get('scale')
                
                if name and file:
                    scale_tuple = None
                    if scale:
                        scale_tuple = self._parse_vector3(scale, (1.0, 1.0, 1.0))
                    
                    self.meshes[name] = MeshInfo(
                        name=name,
                        file=file,
                        scale=scale_tuple
                    )
    
    def _parse_materials(self):
        """Parse material definitions."""
        asset_elem = self.root.find('asset')
        if asset_elem is not None:
            for material_elem in asset_elem.findall('material'):
                name = material_elem.get('name')
                if name:
                    material_info = {}
                    for attr in ['rgba', 'texture', 'texuniform', 'texrepeat', 'reflectance']:
                        value = material_elem.get(attr)
                        if value:
                            material_info[attr] = value
                    self.materials[name] = material_info
    
    def _parse_geom(self, geom_elem: ET.Element) -> GeomInfo:
        """Parse a geometry element."""
        geom_info = GeomInfo()
        
        # Extract mesh reference
        geom_info.mesh_name = geom_elem.get('mesh')
        
        # Extract position
        pos_text = geom_elem.get('pos')
        if pos_text:
            geom_info.position = self._parse_vector3(pos_text)
        
        # Extract quaternion
        quat_text = geom_elem.get('quat')
        if quat_text:
            geom_info.quaternion = self._parse_quaternion(quat_text)
        
        # Extract material and class
        geom_info.material = geom_elem.get('material')
        geom_info.class_ = geom_elem.get('class')
        
        return geom_info
    
    def _parse_joint(self, joint_elem: ET.Element) -> JointInfo:
        """Parse a joint element."""
        joint_name = joint_elem.get('name', 'unnamed_joint')
        joint_type = joint_elem.get('type', 'hinge')
        
        # Parse axis (default to Z-axis for hinge joints)
        axis_text = joint_elem.get('axis', '0 0 1')
        axis = self._parse_vector3(axis_text, (0.0, 0.0, 1.0))
        
        # Parse range
        range_text = joint_elem.get('range')
        joint_range = None
        if range_text:
            try:
                values = [float(x) for x in range_text.strip().split()]
                if len(values) >= 2:
                    joint_range = (values[0], values[1])
            except (ValueError, IndexError):
                pass
        
        # Parse other properties
        damping = None
        damping_text = joint_elem.get('damping')
        if damping_text:
            try:
                damping = float(damping_text)
            except ValueError:
                pass
        
        stiffness = None
        stiffness_text = joint_elem.get('stiffness')
        if stiffness_text:
            try:
                stiffness = float(stiffness_text)
            except ValueError:
                pass
        
        limited = joint_elem.get('limited', 'false').lower() == 'true'
        
        return JointInfo(
            name=joint_name,
            type=joint_type,
            axis=axis,
            range=joint_range,
            damping=damping,
            stiffness=stiffness,
            limited=limited
        )
    
    def _parse_body(self, body_elem: ET.Element, parent_name: Optional[str] = None) -> str:
        """Parse a body element recursively."""
        body_name = body_elem.get('name', 'unnamed_body')
        
        # Parse position and orientation
        pos_text = body_elem.get('pos', '0 0 0')
        position = self._parse_vector3(pos_text)
        
        quat_text = body_elem.get('quat', '1 0 0 0')
        quaternion = self._parse_quaternion(quat_text)
        
        # Create body info
        body_info = BodyInfo(
            name=body_name,
            position=position,
            quaternion=quaternion,
            parent=parent_name
        )
        
        # Parse geometries in this body
        for geom_elem in body_elem.findall('geom'):
            geom_info = self._parse_geom(geom_elem)
            # Only add geometries that reference meshes
            if geom_info.mesh_name:
                body_info.geometries.append(geom_info)
        
        # Parse joint in this body (connects this body to its parent)
        joint_elem = body_elem.find('joint')
        if joint_elem is not None:
            joint_info = self._parse_joint(joint_elem)
            body_info.joint = joint_info
            self.joints[joint_info.name] = joint_info
        
        # Also parse freejoint elements
        freejoint_elem = body_elem.find('freejoint')
        if freejoint_elem is not None:
            joint_name = freejoint_elem.get('name', 'unnamed_freejoint')
            joint_info = JointInfo(
                name=joint_name,
                type='free',
                axis=None,
                range=None
            )
            body_info.joint = joint_info
            self.joints[joint_info.name] = joint_info
        
        # Store this body
        self.bodies[body_name] = body_info
        
        # Parse child bodies
        for child_body_elem in body_elem.findall('body'):
            child_name = self._parse_body(child_body_elem, body_name)
            body_info.children.append(child_name)
        
        return body_name
    
    def _parse_worldbody(self):
        """Parse the worldbody section to extract the body hierarchy."""
        worldbody = self.root.find('worldbody')
        if worldbody is not None:
            # Parse root-level bodies
            for body_elem in worldbody.findall('body'):
                self._parse_body(body_elem, parent_name=None)
    
    def get_mesh_file_path(self, mesh_name: str, meshdir: Optional[str] = None) -> Optional[Path]:
        """Get the full file path for a mesh."""
        if mesh_name not in self.meshes:
            return None
        
        mesh_info = self.meshes[mesh_name]
        
        # Determine mesh directory
        if meshdir is None:
            # Try to get meshdir from compiler element
            compiler_elem = self.root.find('compiler')
            if compiler_elem is not None:
                meshdir = compiler_elem.get('meshdir', 'meshes')
            else:
                meshdir = 'meshes'
        
        # Construct full path
        mesh_file = Path(mesh_info.file)
        if mesh_file.is_absolute():
            return mesh_file
        else:
            return self.mjcf_path.parent / meshdir / mesh_file
    
    def compute_global_transform(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the global transformation (position and rotation matrix) for a body.
        
        Returns:
            Tuple of (global_position, global_rotation_matrix)
        """
        if body_name not in self.bodies:
            return np.zeros(3), np.eye(3)
        
        # Collect the transformation chain from root to this body
        transform_chain = []
        current_name = body_name
        
        while current_name is not None:
            body = self.bodies[current_name]
            transform_chain.append((body.position, body.quaternion))
            current_name = body.parent
        
        # Apply transformations from root to leaf
        transform_chain.reverse()
        
        global_position = np.zeros(3)
        global_rotation = np.eye(3)
        
        for position, quaternion in transform_chain:
            # Convert quaternion to rotation matrix (MJCF uses w,x,y,z format)
            rotation = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])  # Convert to x,y,z,w for scipy
            rotation_matrix = rotation.as_matrix()
            
            # Apply current transformation
            global_position = global_rotation @ np.array(position) + global_position
            global_rotation = global_rotation @ rotation_matrix
        
        return global_position, global_rotation
    
    def get_mesh_transforms_detailed(self) -> Dict[str, List[Tuple[str, np.ndarray, np.ndarray, Optional[str], Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float, float]]]]]:
        """
        Get all mesh transformations with detailed geom info.
        
        Returns:
            Dictionary mapping mesh_name to list of (body_name, global_position, global_rotation_matrix, material, geom_pos, geom_quat)
        """
        mesh_transforms = {}
        
        for body_name, body_info in self.bodies.items():
            for geom_info in body_info.geometries:
                if geom_info.mesh_name:
                    # Compute global transform for the body
                    global_pos, global_rot = self.compute_global_transform(body_name)
                    
                    # Store geom-specific transforms separately
                    geom_pos = geom_info.position
                    geom_quat = geom_info.quaternion
                    
                    # Apply geometry-specific transformation if present
                    if geom_info.position or geom_info.quaternion:
                        geom_pos_array = np.array(geom_info.position) if geom_info.position else np.zeros(3)
                        geom_quat_tuple = geom_info.quaternion if geom_info.quaternion else (1.0, 0.0, 0.0, 0.0)
                        
                        # Convert geometry quaternion to rotation matrix
                        geom_rotation = Rotation.from_quat([geom_quat_tuple[1], geom_quat_tuple[2], geom_quat_tuple[3], geom_quat_tuple[0]])
                        geom_rot_matrix = geom_rotation.as_matrix()
                        
                        # Combine transformations
                        final_pos = global_rot @ geom_pos_array + global_pos
                        final_rot = global_rot @ geom_rot_matrix
                    else:
                        final_pos = global_pos
                        final_rot = global_rot
                    
                    # Add to mesh transforms
                    if geom_info.mesh_name not in mesh_transforms:
                        mesh_transforms[geom_info.mesh_name] = []
                    
                    mesh_transforms[geom_info.mesh_name].append((
                        body_name,
                        final_pos,
                        final_rot,
                        geom_info.material,
                        geom_pos,  # Original geom position
                        geom_quat  # Original geom quaternion
                    ))
        
        return mesh_transforms
    
    def get_mesh_transforms(self) -> Dict[str, List[Tuple[str, np.ndarray, np.ndarray, Optional[str]]]]:
        """
        Get all mesh transformations organized by mesh name.
        
        Returns:
            Dictionary mapping mesh_name to list of (body_name, global_position, global_rotation_matrix, material)
        """
        mesh_transforms = {}
        
        for body_name, body_info in self.bodies.items():
            for geom_info in body_info.geometries:
                if geom_info.mesh_name:
                    # Compute global transform for the body
                    global_pos, global_rot = self.compute_global_transform(body_name)
                    
                    # Apply geometry-specific transformation if present
                    if geom_info.position or geom_info.quaternion:
                        geom_pos = np.array(geom_info.position) if geom_info.position else np.zeros(3)
                        geom_quat = geom_info.quaternion if geom_info.quaternion else (1.0, 0.0, 0.0, 0.0)
                        
                        # Convert geometry quaternion to rotation matrix
                        geom_rotation = Rotation.from_quat([geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]])
                        geom_rot_matrix = geom_rotation.as_matrix()
                        
                        # Combine transformations
                        final_pos = global_rot @ geom_pos + global_pos
                        final_rot = global_rot @ geom_rot_matrix
                    else:
                        final_pos = global_pos
                        final_rot = global_rot
                    
                    # Add to mesh transforms
                    if geom_info.mesh_name not in mesh_transforms:
                        mesh_transforms[geom_info.mesh_name] = []
                    
                    mesh_transforms[geom_info.mesh_name].append((
                        body_name,
                        final_pos,
                        final_rot,
                        geom_info.material
                    ))
        
        return mesh_transforms
    
    def print_hierarchy(self, body_name: Optional[str] = None, indent: int = 0):
        """Print the body hierarchy for debugging."""
        if body_name is None:
            # Print all root bodies
            for name, body in self.bodies.items():
                if body.parent is None:
                    self.print_hierarchy(name, indent)
        else:
            body = self.bodies[body_name]
            joint_desc = ""
            if body.joint:
                joint_desc = f", joint={body.joint.name} ({body.joint.type})"
            print("  " * indent + f"{body_name}: pos={body.position}, quat={body.quaternion}{joint_desc}")
            
            # Print geometries
            for geom in body.geometries:
                geom_desc = f"mesh={geom.mesh_name}"
                if geom.position:
                    geom_desc += f", pos={geom.position}"
                if geom.quaternion:
                    geom_desc += f", quat={geom.quaternion}"
                print("  " * (indent + 1) + f"geom: {geom_desc}")
            
            # Print children
            for child_name in body.children:
                self.print_hierarchy(child_name, indent + 1)
    
    def get_mesh_list(self) -> List[str]:
        """Get a list of all mesh names referenced in the model."""
        mesh_names = set()
        for body_info in self.bodies.values():
            for geom_info in body_info.geometries:
                if geom_info.mesh_name:
                    mesh_names.add(geom_info.mesh_name)
        return sorted(list(mesh_names))
    
    def get_joint_hierarchy(self) -> Dict[str, List[str]]:
        """Get joint hierarchy for skeleton creation."""
        hierarchy = {}
        for body_name, body_info in self.bodies.items():
            if body_info.joint:
                parent_joint = None
                
                # Walk up the body hierarchy to find the nearest ancestor with a joint
                current_parent = body_info.parent
                while current_parent and current_parent in self.bodies:
                    parent_body = self.bodies[current_parent]
                    if parent_body.joint:
                        parent_joint = parent_body.joint.name
                        break
                    current_parent = parent_body.parent
                
                hierarchy[body_info.joint.name] = {
                    'parent_joint': parent_joint,
                    'body_name': body_name,
                    'joint_type': body_info.joint.type,
                    'axis': body_info.joint.axis,
                    'range': body_info.joint.range
                }
        return hierarchy
    
    def get_summary(self) -> Dict[str, int]:
        """Get a summary of the parsed MJCF."""
        return {
            'total_meshes_defined': len(self.meshes),
            'total_bodies': len(self.bodies),
            'total_materials': len(self.materials),
            'total_joints': len(self.joints),
            'meshes_used': len(self.get_mesh_list())
        }


def main():
    """Test the MJCF parser."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mjcf_parser.py <mjcf_file>")
        return
    
    mjcf_file = sys.argv[1]
    parser = MJCFParser(mjcf_file)
    
    print("=== MJCF Parse Summary ===")
    summary = parser.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n=== Body Hierarchy ===")
    parser.print_hierarchy()
    
    print("\n=== Mesh Transforms ===")
    mesh_transforms = parser.get_mesh_transforms()
    for mesh_name, transforms in mesh_transforms.items():
        print(f"\nMesh: {mesh_name}")
        for body_name, pos, rot, material in transforms:
            print(f"  Body: {body_name}")
            print(f"    Position: {pos}")
            print(f"    Rotation: {rot}")
            print(f"    Material: {material}")


if __name__ == "__main__":
    main()
