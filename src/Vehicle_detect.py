from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, weights, device="mps", conf_threshold=0.5, detect_classes=None):
        self.model = YOLO(weights)
        self.device = device
        self.conf_threshold = conf_threshold
        self.detect_classes = detect_classes

        names = self.model.names
        if isinstance(names, dict):
            self.class_names = {int(k): v for k, v in names.items()}
        else:
            self.class_names = {i: name for i, name in enumerate(names)}

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            conf=self.conf_threshold,
            classes=list(self.detect_classes) if self.detect_classes is not None else None,
            verbose=False
        )[0]

        outputs = []

        if results.boxes is None or results.boxes.id is None:
            return outputs

        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            x1, y1, x2, y2 = map(int, box.tolist())
            cx = (x1 + x2) // 2
            foot = (cx, y2)

            outputs.append({
                "track_id": track_id,
                "class_id": class_id,
                "class_name": self.class_names.get(class_id, str(class_id)),
                "bbox": (x1, y1, x2, y2),
                "foot": foot
            })

        return outputs