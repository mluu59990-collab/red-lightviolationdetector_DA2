import cv2
from src.Vehicle_detect import VehicleDetector
from src.tracker import Tracker
from src.traffic_light import TrafficLightDetector


class InferEngine:
    def __init__(self, vehicle_weights, light_weights, video_path):
        self.video_path = video_path

        self.vehicle_detector = VehicleDetector(
            weights=vehicle_weights,
            device="mps",
            conf_threshold=0.5,
            detect_classes={2, 3, 5, 7}   # car, motorcycle, bus, truck
        )

        self.tracker = Tracker(max_age=10)

        self.light_detector = TrafficLightDetector(
            weights=light_weights,
            device="mps",
            conf_threshold=0.5
        )

    def draw_tracked_vehicles(self, frame, tracked_vehicles):
        for obj in tracked_vehicles:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = obj["track_id"]
            class_id = obj["class_id"]
            foot = obj["foot"]

            label = f"ID {track_id} - {class_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.circle(frame, foot, 4, (0, 0, 255), -1)

    def draw_traffic_lights(self, frame, light_detections):
        for obj in light_detections:
            x1, y1, x2, y2 = obj["bbox"]
            label = f'{obj["class_name"]} {obj["conf"]:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    def run(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("Không mở được video")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (960, 540))

            vehicle_detections = self.vehicle_detector.process(frame)
            tracked_vehicles = self.tracker.process(frame, vehicle_detections)
            light_detections = self.light_detector.process(frame)

            self.draw_tracked_vehicles(frame, tracked_vehicles)
            self.draw_traffic_lights(frame, light_detections)

            cv2.imshow("Infer", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = InferEngine(
        vehicle_weights="/Users/Documents/DL-Project/OD_DA2_NN_v1/config/yolov12_best.pt",
        light_weights="/Users/Documents/DL-Project/OD_DA2_NN_v1/config/yolov12_best.pt",
        video_path="/Users/Documents/DL-Project/OD_DA2_NN_v1/data/dataset/7572405326846.mp4"
    )
    engine.run()