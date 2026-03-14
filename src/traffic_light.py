from ultralytics import YOLO


class TrafficLightDetector:
    def __init__(self, weights, device="mps", conf_threshold=0.3):
        self.model = YOLO(weights)
        self.device = device
        self.conf_threshold = conf_threshold

        self.class_names = self.model.names
        if isinstance(self.class_names, dict):
            self.class_names = [self.class_names[i] for i in range(len(self.class_names))]

    def process(self, frame):
        results = self.model(
            frame,
            verbose=False,
            device=self.device,
            conf=self.conf_threshold
        )[0]

        outputs = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            outputs.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "class_id": class_id,
                "class_name": self.class_names[class_id]
            })

        return outputs