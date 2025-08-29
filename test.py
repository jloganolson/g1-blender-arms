#!/usr/bin/env python3
"""
Minimal MuJoCo test to isolate display/XCB issues.
This removes all PyTorch, keyboard input, and other dependencies.
"""

import os

# Try to prevent XCB threading issues
# os.environ['MUJOCO_GL'] = 'egl'
# os.environ['QT_X11_NO_MITSHM'] = '1'
# os.environ['PYTHONUNBUFFERED'] = '1'

import mujoco
import mujoco.viewer as viewer
import numpy as np

# Configuration options
FIXED_PELVIS_MODE = True  # Set to True to fix pelvis in place, False for normal floating base
PELVIS_CONSTRAINT_METHOD = "disable_gravity"  # Options: "disable_gravity", "high_damping", "position_control"


def apply_pelvis_constraints(model, data):
    """Apply constraints to fix the pelvis in place."""
    if not FIXED_PELVIS_MODE:
        return
    
    print(f"Applying pelvis constraint using method: {PELVIS_CONSTRAINT_METHOD}")
    
    if PELVIS_CONSTRAINT_METHOD == "disable_gravity":
        # Method 1: Disable gravity - simplest approach
        # This keeps the robot suspended in air at its initial position
        model.opt.gravity[:] = [0, 0, 0]  # Disable gravity
        print("Gravity disabled - robot will float in place")
    
    elif PELVIS_CONSTRAINT_METHOD == "high_damping":
        # Method 2: Apply very high damping to floating base joint
        floating_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if floating_joint_id >= 0:
            # Floating base has 7 DOFs (3 translation + 4 quaternion)
            # But in terms of velocities/damping, it's 6 DOFs (3 translation + 3 rotation)
            joint_start = model.jnt_dofadr[floating_joint_id]
            model.dof_damping[joint_start:joint_start+6] = 10000.0  # Very high damping
            print("Applied high damping to floating base joint")
    
    elif PELVIS_CONSTRAINT_METHOD == "position_control":
        # Method 3: Store initial position and apply strong position control
        # This will be handled in the simulation loop
        print("Position control mode - will maintain initial pelvis position")


def simple_callback(model=None, data=None):
    """Minimal callback that just loads the model and does basic simulation."""
    # Load the G1 model
    model = mujoco.MjModel.from_xml_path('./g1_description/scene_mjx_alt.xml')
    data = mujoco.MjData(model)
    
    # Apply pelvis constraints if enabled
    apply_pelvis_constraints(model, data)
    
    # Reset to a keyframe (standing pose)
    mujoco.mj_resetDataKeyframe(model, data, 1)
    
    # Set timestep
    model.opt.timestep = 0.005
    
    print("Model loaded successfully!")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Timestep: {model.opt.timestep}")
    print(f"Fixed pelvis mode: {FIXED_PELVIS_MODE}")
    
    return model, data


if __name__ == "__main__":
    print("Starting minimal MuJoCo test...")
    
    try:
        # Launch viewer with minimal callback
        viewer.launch(loader=simple_callback)
    except Exception as e:
        print(f"Error launching viewer: {e}")
        print("\nTrying headless mode instead...")
        
        # Fallback: headless simulation
        model = mujoco.MjModel.from_xml_path('./g1_description/scene_mjx_alt.xml')
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 1)
        
        print("Headless simulation successful!")
        print(f"Model has {model.njnt} joints and {model.nu} actuators")
        
        # Run a few simulation steps
        for i in range(10):
            mujoco.mj_step(model, data)
            
        print("Simulation steps completed without errors.")
