import serial
import time
import pandas as pd
import cv2
import numpy as np
#from matplotlib import pyplot as plt
import os

# Match your Arduino serial port
PORT = 'COM5'
BAUD = 9600
CSV_FILE = 'coordinates_example_6.csv'
radius = 250 #mm

def wait_for_response(ser, expected=None):
    while True:
        line = ser.readline().decode().strip()
        if line:
            print("Arduino:", line)
            if expected is None or line == expected:
                break
            elif line.startswith("ERROR"):
                raise Exception(f"Arduino Error: {line}")

def send_command(ser, command, expect=None):
    print("Sending:", command)
    ser.write((command + '\n').encode())
    wait_for_response(ser, expect)

def get_position(ser):
    ser.write(b'GET_POS\n')
    while True:
        line = ser.readline().decode().strip()
        if line.startswith("POS"):
            _, x_theta, y_phi = line.split()
            return int(x_theta), int(y_phi)
        
def take_photo(cap, x_val1,y_val1):
    #cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)        #faster pictures + 1= second camera plugged 
    #ret, frame = cap.read()

    # Make sure the folder exists
#    os.makedirs("jemoedermap", exist_ok=True)
    if not cap.isOpened():
        print("Failed to open camera.")
        return
    time.sleep(1)
    if not cap.isOpened():
        print("aaahhh")
        return

    # Warm up the camera by capturing a few frames
    for _ in range(10):
        ret, frame = cap.read()

    if ret:
    # Save to that folder
        cv2.imwrite(f"testmaplien/test_{x_val1}_{y_val1}.jpg", frame)
    else:
        print("Failed to capture frame.")
    


def spherical_to_cartesian(theta_deg, phi_deg):
    
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    print(theta_deg)
    print(phi_deg)
    x_val = radius * np.sin(phi) * np.cos(theta)
    y_val = radius * np.sin(phi) * np.sin(theta)
    z_val = radius * np.cos(phi)
    return x_val, y_val, z_val



"""def send_coordinates_from_csv(ser, filename):
    df = pd.read_csv(filename)

    # Assumes columns: Index, X, Y (X in 2nd column, Y in 3rd)
    for i, row in df.iterrows():
        x_theta = int(row[1])
        y_phi= int(row[2])
        #print(np.sin(x_theta))
        #print(np.sin(y_phi))
   
        x_val = round(spherical_to_cartesian(x_theta, y_phi)[0], 0)
        y_val = round(spherical_to_cartesian(x_theta, y_phi)[1], 0)
        z_val = round(spherical_to_cartesian(x_theta, y_phi)[2], 0)

        #print(x_theta)
        #print(y_phi)
        command = f"MOVE_TO {x_theta} {y_phi}"
        send_command(ser, command, expect="MOVE_DONE")
        print(f"Moved to: X={x_theta}°, Y={y_phi}°")
        print(f"Coordinates: {x_val},{y_val},{z_val}")
        #send_command(ser, "LED1ON", expect="LED_1_ON_DONE")
        take_photo(x_val,y_val,z_val)               #take photo and save it to folder
        #time.sleep(2)
        #send_command(ser, "LED1OFF", expect="LED_1_OFF_DONE"       """

def generate_and_send_coordinates(ser):
    x_values = list(range(50, 136, 40))  
    y_values = list(range(0, 361, 30))  
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for y_phi in y_values:
        for x_theta in x_values:
            try:

                x_val = round(spherical_to_cartesian(int(x_theta), int(y_phi))[0], 0)
                y_val = round(spherical_to_cartesian(int(x_theta), int(y_phi))[1], 0)
                z_val = round(spherical_to_cartesian(int(x_theta), int(y_phi))[2], 0)

                command = f"MOVE_TO {x_theta} {y_phi}"
                send_command(ser, command, expect="MOVE_DONE")
                print(f"Moved to: X={x_theta}°, Y={y_phi}°")
                print(f"Coordinates: {x_val},{y_val},{z_val}")

                take_photo(cap, x_theta, y_phi)

            except Exception as e:
                print(f"Error at coordinates ({x_theta}, {y_phi}): {e}")
    cap.release()
    

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # Wait for Arduino 

    try:
        # Step 1: Home the system
        #send_command(ser, "LED1OFF", expect="LED_1_OFF_DONE")
        send_command(ser, "LED1ON", expect="LED_1_ON_DONE")
        #time.sleep(60)
        send_command(ser, "HOME_X", expect="HOME_X_DONE")
        
        # Step 2: Send positions from CSV
        #send_coordinates_from_csv(ser, CSV_FILE)
        generate_and_send_coordinates(ser)

        # Step 3: Get final position
        x_theta, y_phi = get_position(ser)
        print(f"Final Position: X={x_theta}°, Y={y_phi}°")
        #send_command(ser, "LED1ON", expect="LED_1_ON_DONE")
        command = f"MOVE_TO 135 0"
        x_theta, y_phi = get_position(ser)
        send_command(ser, command, expect="MOVE_DONE")
        #command = f"MOVE_TO 0 0"
        #send_command(ser, command, expect="MOVE_DONE")
        # Step 4: Disable motors
        send_command(ser, "DISABLE", expect="STEPPERS_DISABLED")
        send_command(ser, "LED1OFF", expect="LED_1_OFF_DONE")






    except Exception as e:
        print("Error:", e)

    finally:
        ser.close()

if __name__ == "__main__":
    main()
