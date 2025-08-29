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


def simple_callback(model=None, data=None):
    """Minimal callback that just loads the model and does basic simulation."""
    # Load the G1 model
    model = mujoco.MjModel.from_xml_path('./g1_description/scene_mjx_alt.xml')
    data = mujoco.MjData(model)
    
    # Reset to a keyframe (standing pose)
    mujoco.mj_resetDataKeyframe(model, data, 1)
    
    # Set timestep
    model.opt.timestep = 0.005
    
    print("Model loaded successfully!")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Timestep: {model.opt.timestep}")
    
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
