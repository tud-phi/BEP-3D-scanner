# dynamixel_wrapper.py

import time
from dynamixel_sdk import PortHandler, PacketHandler

# --- Default Control Table Addresses ---
# You might need to expand this for more models or allow overrides
ADDR = {
    1.0: {
        "Model_Number": 0,
        "Firmware_Version": 2,
        "ID": 3,
        "Baud_Rate": 4,
        "Return_Delay_Time": 5,
        "CW_Angle_Limit": 6,
        "CCW_Angle_Limit": 8,
        "Temperature_Limit": 11,
        "Min_Voltage_Limit": 12,
        "Max_Voltage_Limit": 13,
        "Max_Torque": 14,
        "Status_Return_Level": 16,
        "Alarm_LED": 17,
        "Shutdown": 18,
        "Torque_Enable": 24,
        "LED": 25,
        "CW_Compliance_Margin": 26,
        "CCW_Compliance_Margin": 27,
        "CW_Compliance_Slope": 28,
        "CCW_Compliance_Slope": 29,
        "Goal_Position": 30,
        "Moving_Speed": 32,
        "Torque_Limit": 34,
        "Present_Position": 36,
        "Present_Speed": 38,
        "Present_Load": 40,
        "Present_Voltage": 42,
        "Present_Temperature": 43,
        "Registered": 44,
        "Moving": 46,
        "Lock": 47,
        "Punch": 48,
    },
    2.0: {
        "Model_Number": 0,
        "Firmware_Version": 6,
        "ID": 7,
        "Baud_Rate": 8,
        "Return_Delay_Time": 9,
        "Drive_Mode": 10,
        "Operating_Mode": 11,
        "Secondary_ID": 12,
        "Protocol_Version": 13, # Protocol Type in docs often
        "Homing_Offset": 20,
        "Moving_Threshold": 24,
        "Temperature_Limit": 31,
        "Max_Voltage_Limit": 32,
        "Min_Voltage_Limit": 34,
        "PWM_Limit": 36,
        "Current_Limit": 38, # Often used for torque control
        "Acceleration_Limit": 40,
        "Velocity_Limit": 44,
        "Max_Position_Limit": 48,
        "Min_Position_Limit": 52,
        "Shutdown": 63,
        "Torque_Enable": 64,
        "LED": 65,
        "Status_Return_Level": 68,
        "Registered_Instruction": 69,
        "Hardware_Error_Status": 70,
        "Velocity_I_Gain": 76,
        "Velocity_P_Gain": 78,
        "Position_D_Gain": 80,
        "Position_I_Gain": 82,
        "Position_P_Gain": 84,
        "Feedforward_2nd_Gain": 88,
        "Feedforward_1st_Gain": 90,
        "Bus_Watchdog": 98,
        "Goal_PWM": 100,
        "Goal_Current": 102,
        "Goal_Velocity": 104,
        "Profile_Acceleration": 108,
        "Profile_Velocity": 112,
        "Goal_Position": 116,
        "Realtime_Tick": 120,
        "Moving": 122,
        "Moving_Status": 123,
        "Present_PWM": 124,
        "Present_Current": 126,
        "Present_Velocity": 128,
        "Present_Position": 132,
        "Velocity_Trajectory": 136,
        "Position_Trajectory": 140,
        "Present_Input_Voltage": 144,
        "Present_Temperature": 146,
    }
}

# --- Data Byte Lengths (Common) ---
# You might need to add more if using less common addresses
LEN = {
    1.0: {
        ADDR[1.0]["Torque_Enable"]: 1,
        ADDR[1.0]["LED"]: 1,
        ADDR[1.0]["Goal_Position"]: 2,
        ADDR[1.0]["Moving_Speed"]: 2,
        ADDR[1.0]["Torque_Limit"]: 2,
        ADDR[1.0]["Present_Position"]: 2,
        ADDR[1.0]["Present_Speed"]: 2,
        ADDR[1.0]["Present_Load"]: 2,
        ADDR[1.0]["Present_Voltage"]: 1,
        ADDR[1.0]["Present_Temperature"]: 1,
        ADDR[1.0]["Moving"]: 1,
    },
    2.0: {
        ADDR[2.0]["Torque_Enable"]: 1,
        ADDR[2.0]["LED"]: 1,
        ADDR[2.0]["Operating_Mode"]: 1,
        ADDR[2.0]["Goal_PWM"]: 2,
        ADDR[2.0]["Goal_Current"]: 2,
        ADDR[2.0]["Goal_Velocity"]: 4,
        ADDR[2.0]["Profile_Acceleration"]: 4,
        ADDR[2.0]["Profile_Velocity"]: 4,
        ADDR[2.0]["Goal_Position"]: 4,
        ADDR[2.0]["Present_PWM"]: 2,
        ADDR[2.0]["Present_Current"]: 2,
        ADDR[2.0]["Present_Velocity"]: 4,
        ADDR[2.0]["Present_Position"]: 4,
        ADDR[2.0]["Present_Input_Voltage"]: 2,
        ADDR[2.0]["Present_Temperature"]: 1,
        ADDR[2.0]["Moving"]: 1,
    }
}


# --- Constants ---
TORQUE_ENABLE_VAL = 1
TORQUE_DISABLE_VAL = 0
COMM_SUCCESS = 0


class DynamixelMotor:
    """A class to wrap Dynamixel SDK functionality for a single motor."""

    def __init__(self, motor_id, port_name, baudrate, protocol_version=2.0, pos_limit_min=0, pos_limit_max=4095):
        """
        Initializes the DynamixelMotor wrapper.

        Args:
            motor_id (int): The ID of the Dynamixel motor.
            port_name (str): The serial port name (e.g., '/dev/ttyUSB0', 'COM3').
            baudrate (int): The communication baudrate.
            protocol_version (float): The Dynamixel protocol version (1.0 or 2.0).
            pos_limit_min (int): Minimum operational position value.
            pos_limit_max (int): Maximum operational position value.
        """
        self.motor_id = motor_id
        self.port_name = port_name
        self.baudrate = baudrate
        self.protocol_version = float(protocol_version)

        if self.protocol_version not in ADDR:
            raise ValueError(f"Unsupported protocol version: {protocol_version}. Supported: {list(ADDR.keys())}")

        self.addr = ADDR[self.protocol_version]
        self.len = LEN[self.protocol_version] # Get lengths for this protocol

        self.pos_limit_min = pos_limit_min
        self.pos_limit_max = pos_limit_max

        self.portHandler = PortHandler(self.port_name)
        self.packetHandler = PacketHandler(self.protocol_version)

        self.is_connected = False
        self.is_torque_enabled = False

        print(f"Dynamixel Wrapper Initialized: ID {self.motor_id}, Port {self.port_name}, Baud {self.baudrate}, Protocol {self.protocol_version}")

    def _check_comm_result(self, result, error, action_msg="Operation"):
        """Internal helper to check communication results and print status."""
        if result != COMM_SUCCESS:
            print(f"!!! {action_msg} failed: {self.packetHandler.getTxRxResult(result)}")
            return False
        elif error != 0:
            print(f"!!! {action_msg} error: {self.packetHandler.getRxPacketError(error)}")
            return False
        # print(f"    {action_msg} successful.") # Optional: reduce verbosity
        return True

    def connect(self):
        """Opens the serial port and sets the baudrate."""
        if self.is_connected:
            print("Already connected.")
            return True
        if not self.portHandler.openPort():
            print(f"!!! Failed to open port {self.port_name}")
            return False
        print(f"Port {self.port_name} opened.")

        if not self.portHandler.setBaudRate(self.baudrate):
            print(f"!!! Failed to set baudrate to {self.baudrate}")
            self.portHandler.closePort()
            return False
        print(f"Baudrate set to {self.baudrate}.")

        self.is_connected = True
        return True

    def disconnect(self):
        """Disables torque (if enabled) and closes the serial port."""
        if not self.is_connected:
            print("Not connected.")
            return True
        if self.is_torque_enabled:
            self.disable_torque() # Attempt to disable torque before closing
        self.portHandler.closePort()
        self.is_connected = False
        self.is_torque_enabled = False
        print(f"Port {self.port_name} closed.")
        return True

    def ping(self):
        """Pings the motor to check communication."""
        if not self.is_connected:
            print("!!! Cannot ping: Not connected.")
            return False
        model_number, result, error = self.packetHandler.ping(self.portHandler, self.motor_id)
        success = self._check_comm_result(result, error, f"Ping ID {self.motor_id}")
        if success:
            print(f"    Ping Success! Model Number: {model_number}")
        return success

    def reboot(self):
        """Sends a reboot command to the motor (Protocol 2.0 only)."""
        if not self.is_connected:
            print("!!! Cannot reboot: Not connected.")
            return False
        if self.protocol_version != 2.0:
            print("!!! Reboot command only available for Protocol 2.0")
            return False

        result, error = self.packetHandler.reboot(self.portHandler, self.motor_id)
        success = self._check_comm_result(result, error, f"Reboot ID {self.motor_id}")
        if success:
            print("    Motor rebooting. Wait a few seconds before sending further commands.")
            time.sleep(2.0) # Give time for reboot
            self.is_torque_enabled = False # Torque is disabled after reboot
        return success

    def _write_data(self, address, data):
        """Internal helper for writing data based on length defined in LEN."""
        if address not in self.len:
            print(f"!!! Write Error: Address {address} not found in known lengths for protocol {self.protocol_version}.")
            return False

        length = self.len[address]
        if length == 1:
            result, error = self.packetHandler.write1ByteTxRx(self.portHandler, self.motor_id, address, int(data))
        elif length == 2:
            result, error = self.packetHandler.write2ByteTxRx(self.portHandler, self.motor_id, address, int(data))
        elif length == 4:
            result, error = self.packetHandler.write4ByteTxRx(self.portHandler, self.motor_id, address, int(data))
        else:
            print(f"!!! Write Error: Unsupported data length {length} for address {address}")
            return False

        # Get the key name for the address for better error messages
        addr_name = next((name for name, addr in self.addr.items() if addr == address), f"Address {address}")
        return self._check_comm_result(result, error, f"Write {addr_name} ({data})")

    def _read_data(self, address):
        """Internal helper for reading data based on length defined in LEN."""
        if address not in self.len:
            print(f"!!! Read Error: Address {address} not found in known lengths for protocol {self.protocol_version}.")
            return None # Indicate failure

        length = self.len[address]
        if length == 1:
            value, result, error = self.packetHandler.read1ByteTxRx(self.portHandler, self.motor_id, address)
        elif length == 2:
            value, result, error = self.packetHandler.read2ByteTxRx(self.portHandler, self.motor_id, address)
        elif length == 4:
            value, result, error = self.packetHandler.read4ByteTxRx(self.portHandler, self.motor_id, address)
        else:
            print(f"!!! Read Error: Unsupported data length {length} for address {address}")
            return None # Indicate failure

        # Get the key name for the address for better error messages
        addr_name = next((name for name, addr in self.addr.items() if addr == address), f"Address {address}")
        if self._check_comm_result(result, error, f"Read {addr_name}"):
            return value
        else:
            return None # Indicate failure

    def enable_torque(self):
        """Enables torque for the motor."""
        if not self.is_connected:
            print("!!! Cannot enable torque: Not connected.")
            return False
        print(f"Enabling Torque for ID {self.motor_id}...")
        if self._write_data(self.addr["Torque_Enable"], TORQUE_ENABLE_VAL):
            self.is_torque_enabled = True
            return True
        self.is_torque_enabled = False # Ensure state reflects reality on failure
        return False

    def disable_torque(self):
        """Disables torque for the motor."""
        if not self.is_connected:
            print("!!! Cannot disable torque: Not connected.")
            # If not connected, torque is effectively disabled anyway
            self.is_torque_enabled = False
            return True # Return true as the state matches desired state (off)
        print(f"Disabling Torque for ID {self.motor_id}...")
        if self._write_data(self.addr["Torque_Enable"], TORQUE_DISABLE_VAL):
            self.is_torque_enabled = False
            return True
        # Don't know the state for sure if write fails
        return False

    def set_goal_position(self, position):
        """Sets the goal position, clamping within defined limits."""
        if not self.is_connected:
            print("!!! Cannot set goal position: Not connected.")
            return False
        if not self.is_torque_enabled:
            print("!!! Warning: Torque is not enabled. Motor will not move to goal position.")
            # Allow setting goal even if torque is off, but warn

        clamped_position = max(self.pos_limit_min, min(int(position), self.pos_limit_max))
        if clamped_position != position:
             print(f"    Position {position} clamped to {clamped_position} (Limits: {self.pos_limit_min}-{self.pos_limit_max})")

        #print(f"Setting Goal Position: {clamped_position}") # Reduce verbosity
        return self._write_data(self.addr["Goal_Position"], clamped_position)

    def get_present_position(self):
        """Reads and returns the current position."""
        if not self.is_connected:
            print("!!! Cannot get present position: Not connected.")
            return None
        return self._read_data(self.addr["Present_Position"])

    def set_led(self, state):
        """Turns the motor's LED on (True) or off (False)."""
        if not self.is_connected:
            print("!!! Cannot set LED: Not connected.")
            return False
        led_val = 1 if state else 0
        return self._write_data(self.addr["LED"], led_val)

    def get_present_load(self):
        """Reads and returns the current load (Protocol 1.0)."""
         # Protocol 1.0 specific - may need adjustment based on motor model's interpretation
        if self.protocol_version != 1.0:
            print("!!! get_present_load is typically for Protocol 1.0. Use get_present_current for Protocol 2.0.")
            return None
        if not self.is_connected:
            print("!!! Cannot get present load: Not connected.")
            return None
        load_raw = self._read_data(self.addr["Present_Load"])
        if load_raw is None:
            return None
        # Convert raw load value (0-1023 negative, 1024-2047 positive) to percentage/direction
        if load_raw > 1023:
            load_percent = (load_raw - 1023) / 1023.0 * 100.0 # CCW Torque
            direction = "CCW"
        else:
            load_percent = load_raw / 1023.0 * 100.0 # CW Torque
            direction = "CW"
        return load_percent, direction # Returns tuple: (percentage, direction_string)


    def get_present_current(self):
        """Reads and returns the present current (Protocol 2.0)."""
        if self.protocol_version != 2.0:
            print("!!! get_present_current is typically for Protocol 2.0. Use get_present_load for Protocol 1.0.")
            return None
        if not self.is_connected:
            print("!!! Cannot get present current: Not connected.")
            return None
        # Note: Value is often in mA. Check your motor's e-manual.
        return self._read_data(self.addr["Present_Current"])


    def set_operating_mode(self, mode):
        """Sets the operating mode (Protocol 2.0 only)."""
        # Common Modes for Protocol 2.0 (Check E-Manual for your specific motor):
        # 0: Current Control Mode
        # 1: Velocity Control Mode
        # 3: Position Control Mode (Default)
        # 4: Extended Position Control Mode (Multi-turn)
        # 5: Current-based Position Control Mode
        # 16: PWM Control Mode (Voltage Control)
        if self.protocol_version != 2.0:
            print("!!! Operating Mode setting is only available for Protocol 2.0")
            return False
        if not self.is_connected:
            print("!!! Cannot set operating mode: Not connected.")
            return False

        # Important: Torque must be OFF to change operating mode
        needs_torque_reenable = False
        if self.is_torque_enabled:
             print("    Disabling torque temporarily to change operating mode...")
             if self.disable_torque():
                 needs_torque_reenable = True
             else:
                 print("!!! Failed to disable torque. Cannot change operating mode.")
                 return False
             time.sleep(0.1) # Short delay after disabling torque

        print(f"Setting Operating Mode to {mode}...")
        success = self._write_data(self.addr["Operating_Mode"], mode)

        if needs_torque_reenable:
            print("    Re-enabling torque...")
            if not self.enable_torque():
                print("!!! Warning: Failed to re-enable torque after changing operating mode.")
            time.sleep(0.1) # Short delay

        return success

    def get_operating_mode(self):
        """Gets the current operating mode (Protocol 2.0 only)."""
        if self.protocol_version != 2.0:
            print("!!! Operating Mode reading is only available for Protocol 2.0")
            return None
        if not self.is_connected:
            print("!!! Cannot get operating mode: Not connected.")
            return None
        return self._read_data(self.addr["Operating_Mode"])
