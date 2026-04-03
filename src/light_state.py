class TrafficLightStateManager:
    def __init__(self):
        self.last_detections = []

    def update(self, detections):
        self.last_detections = detections

    def get_detections(self):
        return self.last_detections

    def is_red(self):
        for obj in self.last_detections:
            class_name = obj["class_name"].lower()
            if "red" in class_name:
                return True
        return False