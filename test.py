from ultralytics import YOLO
model = YOLO("best.pt")
model("test.jpg",show = True)