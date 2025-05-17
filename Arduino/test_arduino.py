import serial
import serial.tools.list_ports
import time

# Update this to match your port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux)
PORT = '/dev/ttyS0'  # or 'COMx' on Windows
BAUD = 9600

# Connect to Arduino
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # Wait for Arduino to reset

# Turn LED on
ser.write(b'1')
print("LED ON")
time.sleep(2)

# Turn LED off
ser.write(b'0')
print("LED OFF")

ser.close()
