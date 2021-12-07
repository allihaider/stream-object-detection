import cv2
from collections import deque

frame_queue = deque(maxlen=30)

stream_address = "http://localhost:8080"
cap = cv2.VideoCapture(stream_address)

if (cap.isOpened() == False):
        print('Could not open camera')

while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            print("Frame read")
        else:
            print("Frame not read")

