from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, weights, device="mps", conf_threshold=0.3, detect_classes=None):
        self.model = YOLO(weights)
        self.device = device
        self.conf_threshold = conf_threshold
        self.detect_classes = detect_classes

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

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            if conf < self.conf_threshold:
                continue
            if self.detect_classes is not None and class_id not in self.detect_classes:
                continue

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "class_id": class_id,
                "class_name": self.class_names[class_id]
            })

        return detections