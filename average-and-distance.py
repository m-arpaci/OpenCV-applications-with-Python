import cv2
import numpy as np
import math

#created by @m-arpaci

real_distance = 10  # Outer diameter of the ring
                    # If the object is not a ring, 
                    # its horizontal distance must be entered.

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Green color range in HSV
    lowerLimit = np.array([hue - 20, 100, 100], dtype=np.uint8)
    upperLimit = np.array([hue + 20, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

def get_distance(w, real_distance):
    horizontal_angle = 55  # Camera horizontal angle
    view_angle = horizontal_angle * w / 640  # 640 = Camera horizontal pixel count
    radyan_angle = math.radians(view_angle / 2)
    tan = math.tan(radyan_angle)
    distance = (real_distance / 2) / tan
    return distance

green = [0, 255, 0]  # Green in BGR format
lowerLimit, upperLimit = get_limits(green)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirroring the image

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=green)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
        # x: The x coordinate of the top-left corner of the bounding box.
        # y: The y coordinate of the top-left corner of the bounding box.
        # w: The width of the bounding box.
        # h: The height of the bounding box.

        if 310 < x + w / 2 < 330:  # Average range
            print("centered")
        if 45 < get_distance(w, real_distance) < 50:  # Desired distance range
            print("distance okay")
    else:
        print("No object")

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
