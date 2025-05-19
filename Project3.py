import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with your custom-trained model if available

# Function to detect and count products in a frame
def detect_and_count(frame):
    results = model(frame)
    detections = results[0]

    counts = defaultdict(int)
    annotated_frame = detections.plot()

    for box in detections.boxes:
        class_id = int(box.cls[0])
        class_name = detections.names[class_id]
        counts[class_name] += 1

    return annotated_frame, counts

# Choose source: image or video
USE_IMAGE = True  # Set to False to use webcam or video

if USE_IMAGE:
    # Load and process an image
    image = cv2.imread("nutella.jpg")
    annotated_img, product_counts = detect_and_count(image)

    # Show results
    print("Detected Products:")
    for product, count in product_counts.items():
        print(f"{count} {product}(s)")

    cv2.imshow("Retail Shelf Inventory", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # Use video capture (webcam or video file)
    cap = cv2.VideoCapture(0)  # Replace 0 with video file path if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, product_counts = detect_and_count(frame)

        # Print product counts in real-time (optional)
        print("Live Inventory:", dict(product_counts))

        cv2.imshow("Live Shelf Inventory", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
