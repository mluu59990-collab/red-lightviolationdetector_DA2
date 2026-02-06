from ultralytics import YOLO
#Lay model
model = YOLO("yolov8n.pt")
#sd model
model.train(data = "/kaggle/working/ALLCAM1TO6-2/data.yaml",
            epochs = 50,
            imgsz = 640,
            batch = 8,
            workers= 2,
            device= "cuda")
