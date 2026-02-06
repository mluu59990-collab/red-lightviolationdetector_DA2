from ultralytics import YOLO
#Lay model
model = YOLO("yolov8n.pt")
#sd model
model.train(data = "/Users/Documents/DL-Project/OD_DA2_NN_v1/data/dataset/ALLCAM1TO6.v2i.yolov8/data.yaml",
            epochs = 50,
            imgsz = 640,
            batch = 8,
            workers= 2,
            device= "cuda")
