#!/usr/bin/env python3
"""
Export MJCF joint information to JSON for use in Blender.
This exports joint limits, axes, and other properties.
"""

import json
from pathlib import Path
from mjcf_parser import MJCFParser


def export_joint_info(mjcf_path: str = "./g1_description/g1_mjx_alt.xml",
                     output_path: str = "output/mjcf_joint_info.json") -> None:
    """Export joint information from MJCF to JSON."""
    parser = MJCFParser(mjcf_path)
    
    joint_info = {}
    
    for joint_name, joint_data in parser.joints.items():
        info = {
            "type": joint_data.type,
            "axis": joint_data.axis,
            "limited": joint_data.limited
        }
        
        if joint_data.range:
            info["range"] = joint_data.range
            
        if joint_data.damping is not None:
            info["damping"] = joint_data.damping
            
        if joint_data.stiffness is not None:
            info["stiffness"] = joint_data.stiffness
            
        joint_info[joint_name] = info
        
    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(joint_info, f, indent=2)
        
    print(f"Exported joint information to: {output_path}")
    print(f"Total joints: {len(joint_info)}")
    
    # Print summary
    print("\nJoint Summary:")
    for name, info in joint_info.items():
        range_str = f"range: {info.get('range', 'unlimited')}" if info.get('limited') else "unlimited"
        print(f"  {name}: {info['type']} joint, {range_str}")


if __name__ == "__main__":
    export_joint_info()
