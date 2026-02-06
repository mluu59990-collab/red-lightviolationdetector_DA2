from ultralytics import YOLO
import cv2
model = YOLO("best.py")
video_path = ""
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps= int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width,height)
)
while True:
    ret,frame = cap.read()
    if not ret:
        break
    results = model(frame,conf= 0.5)
    annotated = results[0].plot()
    #luu video
    out.write(annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindow()
