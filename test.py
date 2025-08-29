#!/usr/bin/env python3
"""
Clean G1 Robot Test with Unitree-style Elastic Band Suspension
"""

import os
import time
import logging

# Try to prevent XCB threading issues
os.environ['MUJOCO_GL'] = 'egl'
os.environ['QT_X11_NO_MITSHM'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import mujoco
import mujoco.viewer as viewer
import numpy as np


# Configuration options
USE_ELASTIC_BAND = True       # Toggle elastic band suspension
AUTO_TEST_DURATION = None     # Seconds to run test (None = run indefinitely)
JOINT_MOVEMENT_AMPLITUDE = 0.1  # How much to move joints during test
TARGET_HEIGHT = 1.2           # Suspension height above ground

# Elastic Band Constants
ELASTIC_BAND_STIFFNESS = 1000.0   # N/m - Spring stiffness
ELASTIC_BAND_DAMPING = 100.0      # Ns/m - Damping coefficient
ELASTIC_BAND_VERTICAL_MULTIPLIER = 2.0    # Vertical stiffness multiplier
ELASTIC_BAND_VERTICAL_DAMPING_MULTIPLIER = 1.5  # Vertical damping multiplier

# Logging Configuration
LOG_LEVEL = logging.INFO      # Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                                  # DEBUG: Shows pelvis position every 50 steps + all other messages
                                  # INFO: Shows startup, completion, and result messages (recommended)
                                  # WARNING: Shows only warnings and errors
                                  # ERROR: Shows only errors
                                  # CRITICAL: Shows only critical errors


class ElasticBand:
    """Elastic band suspension system based on Unitree's implementation."""
    
    def __init__(self, stiffness=ELASTIC_BAND_STIFFNESS, damping=ELASTIC_BAND_DAMPING, target_height=TARGET_HEIGHT):
        self.enable = True
        self.stiffness = np.array([stiffness, stiffness, stiffness * ELASTIC_BAND_VERTICAL_MULTIPLIER])  # Stronger vertical
        self.damping = np.array([damping, damping, damping * ELASTIC_BAND_VERTICAL_DAMPING_MULTIPLIER])       # Higher vertical damping
        self.target_pos = np.array([0.0, 0.0, target_height])
        
    def Advance(self, position, velocity):
        """Calculate elastic band forces similar to Unitree implementation."""
        if not self.enable:
            return np.zeros(3)
            
        # Calculate displacement from target position
        displacement = position - self.target_pos
        
        # Spring force: F = -k * displacement - c * velocity
        force = -self.stiffness * displacement - self.damping * velocity
        
        return force


def setup_logging():
    """Configure logging with the specified verbosity level."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def log_pelvis_position(data, step_count, logger):
    """Log pelvis position for monitoring stability."""
    if step_count % 50 == 0:  # Log every 50 steps (about 0.25 seconds)
        pelvis_pos = data.qpos[0:3]
        target_pos = np.array([0.0, 0.0, TARGET_HEIGHT])
        drift = np.linalg.norm(pelvis_pos - target_pos)
        
        logger.debug(f"Step {step_count:4d}: Pelvis pos=[{pelvis_pos[0]:6.3f}, {pelvis_pos[1]:6.3f}, {pelvis_pos[2]:6.3f}] "
                    f"drift={drift:6.4f}m")


def animate_joints(data, time_elapsed, elastic_band=None, band_attached_link=None):
    """Move joints and apply elastic band forces exactly like Unitree."""
    # Apply elastic band forces exactly like Unitree implementation
    if USE_ELASTIC_BAND and elastic_band is not None and band_attached_link is not None:
        if elastic_band.enable:
            # Exact Unitree implementation: apply force using first 3 elements of qpos and qvel
            data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                data.qpos[:3], data.qvel[:3]
            )
    
    # Create sinusoidal movements for arm joints to test stability
    freq = 0.5  # Hz
    
    # Find actuator indices (which control the joints)
    actuator_names = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint"
    ]
    
    for i, actuator_name in enumerate(actuator_names):
        actuator_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id >= 0:
            # Create phase-shifted sinusoidal movement
            phase = i * np.pi / 3  # Different phase for each joint
            data.ctrl[actuator_id] = JOINT_MOVEMENT_AMPLITUDE * np.sin(2 * np.pi * freq * time_elapsed + phase)


def setup_model(logger):
    """Load and configure the G1 model."""
    # Load the G1 model
    model = mujoco.MjModel.from_xml_path('./g1_description/scene_mjx_alt.xml')
    data = mujoco.MjData(model)
    
    # Reset to a keyframe (standing pose)
    mujoco.mj_resetDataKeyframe(model, data, 1)
    
    # Lift robot to suspension height if using elastic band
    if USE_ELASTIC_BAND:
        data.qpos[2] = TARGET_HEIGHT
    
    # Set timestep
    model.opt.timestep = 0.005
    
    logger.info("=" * 60)
    logger.info("G1 Robot - Elastic Band Suspension Test")
    logger.info("=" * 60)
    logger.info(f"Elastic band enabled: {USE_ELASTIC_BAND}")
    if USE_ELASTIC_BAND:
        logger.info(f"  Stiffness: {ELASTIC_BAND_STIFFNESS} N/m")
        logger.info(f"  Damping: {ELASTIC_BAND_DAMPING} Ns/m")
        logger.info(f"  Vertical multipliers: {ELASTIC_BAND_VERTICAL_MULTIPLIER}x stiffness, {ELASTIC_BAND_VERTICAL_DAMPING_MULTIPLIER}x damping")
    logger.info(f"Target height: {TARGET_HEIGHT}m")
    logger.info(f"Test duration: {'Infinite (until viewer closed)' if AUTO_TEST_DURATION is None else f'{AUTO_TEST_DURATION} seconds'}")
    logger.info(f"Joint movement amplitude: {JOINT_MOVEMENT_AMPLITUDE}")
    logger.info(f"Number of joints: {model.njnt}")
    logger.info(f"Number of actuators: {model.nu}")
    logger.info("-" * 60)
    
    return model, data


if __name__ == "__main__":
    # Setup logging first
    logger = setup_logging()
    logger.info("Starting G1 Robot Elastic Band Test...")
    
    try:
        # Set up model
        model, data = setup_model(logger)
        
        # Create elastic band and find attachment point exactly like Unitree
        elastic_band = None
        band_attached_link = None
        if USE_ELASTIC_BAND:
            elastic_band = ElasticBand(stiffness=ELASTIC_BAND_STIFFNESS, damping=ELASTIC_BAND_DAMPING, target_height=TARGET_HEIGHT)
            # Exact Unitree G1 implementation: attach to torso_link
            band_attached_link = model.body("torso_link").id
            logger.info(f"✓ Elastic band created, attached to torso_link (id: {band_attached_link})")
        
        # Launch viewer exactly like Unitree
        with mujoco.viewer.launch_passive(model, data) as viewer_instance:
            logger.info("✓ Viewer launched successfully")
            start_time = time.time()
            step_count = 0
            
            while viewer_instance.is_running() and (AUTO_TEST_DURATION is None or (time.time() - start_time) < AUTO_TEST_DURATION):
                # Apply elastic band forces first (before mj_step, like Unitree)
                if USE_ELASTIC_BAND and elastic_band.enable:
                    data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                        data.qpos[:3], data.qvel[:3]
                    )
                
                # Step simulation
                mujoco.mj_step(model, data)
                
                # Animate joints
                animate_joints(data, data.time, elastic_band, band_attached_link)
                
                # Log pelvis position
                log_pelvis_position(data, step_count, logger)
                
                # Sync with viewer
                viewer_instance.sync()
                
                step_count += 1
                time.sleep(0.001)  # Small delay to prevent excessive CPU usage
            
            # Final results
            elapsed_time = time.time() - start_time
            if AUTO_TEST_DURATION is None:
                logger.info(f"\n✓ Test completed after {elapsed_time:.2f} seconds (viewer closed)")
            else:
                logger.info(f"\n✓ Test completed after {AUTO_TEST_DURATION} seconds")
            if USE_ELASTIC_BAND:
                final_pos = data.qpos[0:3]
                target_pos = np.array([0.0, 0.0, TARGET_HEIGHT])
                final_drift = np.linalg.norm(final_pos - target_pos)
                logger.info(f"Final pelvis position: [{final_pos[0]:6.3f}, {final_pos[1]:6.3f}, {final_pos[2]:6.3f}]")
                logger.info(f"Total drift from target: {final_drift:6.4f}m")
                
                if final_drift < 0.01:
                    logger.info("✓ EXCELLENT: Very stable suspension")
                elif final_drift < 0.05:
                    logger.info("✓ GOOD: Stable suspension")
                else:
                    logger.warning("⚠ WARNING: Some drift detected")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error("Check your display connection and MuJoCo installation.")