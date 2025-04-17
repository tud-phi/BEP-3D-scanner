#!/usr/bin/env python
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Dynamixel Keyboard Control using Wrapper Class
#
# Controls a Dynamixel motor position using 'j' (CW) and 'l' (CCW) keys,
# utilizing the DynamixelMotor wrapper class.
# Displays a simple text visualization of the motor's current position.
# Press 'q' or ESC to quit.
#
# Requirements: pip install dynamixel-sdk getch
#               Place dynamixel_wrapper.py in the same directory or Python path.
#------------------------------------------------------------------------------

import sys
import time
from getch import getch # Cross-platform getch

# Import the wrapper class
from dynamixel_motor import DynamixelMotor

# --- Configuration Section (!!! MUST BE UPDATED FOR YOUR SETUP !!!) ---
PROTOCOL_VERSION        = 2.0           # Dynamixel protocol version (1.0 or 2.0)
DXL_ID                  = 1             # Dynamixel ID: Check your motor configuration
# BAUDRATE                = 57600         # Dynamixel baudrate: Check your motor config
BAUDRATE                = 1000000         # Dynamixel baudrate: Check your motor config
DEVICENAME              = '/dev/ttyUSB0' # Linux: /dev/ttyUSB0, /dev/ttyACM0 | Windows: COM1, COM3 etc. | Mac: /dev/tty.usbserial-*

# -- Position Limits & Step ---
# Define the operational range for *your* motor and setup.
# This will be passed to the wrapper for clamping.
DXL_MINIMUM_POSITION_VALUE = 0           # Example: 0
DXL_MAXIMUM_POSITION_VALUE = 4095        # Example: 4095 for Protocol 2.0, 1023 for Protocol 1.0
POSITION_STEP_SIZE      = 50            # Amount to change goal position per key press

# -- Visualization ---
VISUALIZATION_WIDTH     = 50            # Width of the text visualization bar
# ----------------------- End of Configuration -----------------------

# --- Helper Function (Kept in this script for display purposes) ---
def visualize_position(current_pos, min_pos, max_pos, width):
    """Prints a simple text-based visualization of the position."""
    if current_pos is None: # Handle read errors
        sys.stdout.write("\rPosition: READ_ERR" + " " * (width + 15)) # Clear rest of line
        sys.stdout.flush()
        return

    # Clamp for visualization purposes, even though the wrapper also clamps
    display_pos = max(min_pos, min(current_pos, max_pos))

    if max_pos == min_pos: # Avoid division by zero
        position_ratio = 0
    else:
        position_ratio = (display_pos - min_pos) / (max_pos - min_pos)

    marker_pos = int(position_ratio * (width - 1)) # -1 for 0-based index

    vis_bar = ['-'] * width
    vis_bar[marker_pos] = '*'

    # Use sys.stdout.write and \r to overwrite the line
    sys.stdout.write(f"\rPosition: {current_pos:04d} |{min_pos}|{''.join(vis_bar)}|{max_pos}|  ")
    sys.stdout.flush()

# --- Main Execution ---
def main():
    # Instantiate the motor wrapper
    motor = DynamixelMotor(
        motor_id=DXL_ID,
        port_name=DEVICENAME,
        baudrate=BAUDRATE,
        protocol_version=PROTOCOL_VERSION,
        pos_limit_min=DXL_MINIMUM_POSITION_VALUE,
        pos_limit_max=DXL_MAXIMUM_POSITION_VALUE
    )

    # Connect to the motor
    if not motor.connect():
        print("Failed to connect to the motor. Exiting.")
        return # Exit if connection fails

    # Optional: Ping the motor to verify communication
    if not motor.ping():
         print("Motor did not respond to ping. Continuing anyway...")
         # Decide if you want to exit here or try to continue

    # Enable torque
    if not motor.enable_torque():
        print("Failed to enable torque. Exiting.")
        motor.disconnect() # Clean up connection
        return

    # Get initial position
    present_pos = motor.get_present_position()
    if present_pos is None:
        print("Failed to read initial position. Setting goal to midpoint.")
        # Calculate midpoint based on limits passed to constructor
        goal_position = int((motor.pos_limit_max + motor.pos_limit_min) / 2)
    else:
        goal_position = present_pos # Start with current position as goal
        print(f"Initial position: {goal_position}")


    print("\n--- Motor Control (Using Wrapper) ---")
    print("  j: Move Clockwise")
    print("  l: Move Counter-Clockwise")
    print("  q: Quit and Disable Torque")
    print("-------------------------------------\n")

    try:
        while True:
            # Get current position for display
            present_pos = motor.get_present_position()

            # Update visualization
            visualize_position(present_pos, motor.pos_limit_min, motor.pos_limit_max, VISUALIZATION_WIDTH)

            # Get keyboard input
            key = getch() # Read one character

            new_goal_set = False
            if key == 'j':
                goal_position += POSITION_STEP_SIZE
                new_goal_set = True
            elif key == 'l':
                goal_position -= POSITION_STEP_SIZE
                new_goal_set = True
            elif key == 'q' or key == '\x1b': # 'q' or ESC key
                print("\nQuitting...")
                break # Exit loop

            # If a movement key was pressed, set the new goal position via the wrapper
            if new_goal_set:
                # The wrapper's set_goal_position method handles clamping
                if not motor.set_goal_position(goal_position):
                    # Optional: Handle write failure if needed, though wrapper prints errors
                    print("!!! Failed to set goal position.")
                # Small delay helps prevent overwhelming the serial port and allows movement start
                time.sleep(0.02) # Slightly reduced delay might be okay

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Exiting.")
    finally:
        # Ensure disconnection happens cleanly
        print("Disconnecting motor...")
        motor.disconnect()
        print("Motor disconnected.")

if __name__ == "__main__":
    main()
