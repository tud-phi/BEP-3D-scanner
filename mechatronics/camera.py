import cv2
from matplotlib import pyplot as plt
import os

#cap = cv2.VideoCapture(0)
#ret, frame = cap.read()
#plt.imshow(frame)


def take_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # Make sure the folder exists
    os.makedirs("jemoedermap", exist_ok=True)

    # Save to that folder
    cv2.imwrite("jemoeder.jpg", frame)
    cap.release()
    
take_photo()