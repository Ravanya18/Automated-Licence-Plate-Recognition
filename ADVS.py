import cv2
import numpy as np

INPUT_VIDEO = "webcam.mp4"
OUTPUT_VIDEO = "output_annotated.avi"
SAVE_OUTPUT = True

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    decision = "Searching..."
    area = 0
    cx = -1

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 500:
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2
            frame_center = frame.shape[1] // 2

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if cx < frame_center - 50:
                decision = "Turning LEFT"
            elif cx > frame_center + 50:
                decision = "Turning RIGHT"
            else:
                decision = "Target CENTERED"

            if area > 10000:
                decision = "Landing - Target Close"

    print(decision)
    cv2.putText(frame, f"Action: {decision}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Autonomous Drone Vision", frame)

    if SAVE_OUTPUT:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # this is now correctly inside the loop

cap.release()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()
