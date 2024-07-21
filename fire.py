from ultralytics import YOLO
import cvzone
import cv2
import math
from picamera2 import PiCamera2

# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Running real time from webcam
cap = cv2.VideoCapture(0)
model = YOLO("best.pt")


# Reading the classes
classnames = ["fire"]

while True:
    # ret, frame = cap.read()
    # frame = cv2.resize(frame, (640, 480))
    # result = model(frame, stream=True)
    #
    # # Getting bbox,confidence and class names informations to work with
    # for info in result:
    #     boxes = info.boxes
    #     for box in boxes:
    #         confidence = box.conf[0]
    #         confidence = math.ceil(confidence * 100)
    #         Class = int(box.cls[0])
    #         if confidence > 50:
    #             x1, y1, x2, y2 = box.xyxy[0]
    #             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
    #             cvzone.putTextRect(
    #                 frame,
    #                 f"{classnames[Class]} {confidence}%",
    #                 [x1 + 8, y1 + 100],
    #                 scale=1.5,
    #                 thickness=2,
    #             )
    frame = picam2.capture_array()
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Camera", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break
