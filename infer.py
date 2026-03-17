import cv2
from src.Vehicle_detect import VehicleDetector
from src.traffic_light import TrafficLightDetector


class InferEngine:
    def __init__(self, vehicle_weights, light_weights, video_path):
        self.video_path = video_path
        self.frame_idx = 0
        self.last_light_detections = []

        self.vehicle_detector = VehicleDetector(
            weights=vehicle_weights,
            device="mps",
            conf_threshold=0.5,
            detect_classes={2, 3, 5, 7}
        )

        self.light_detector = TrafficLightDetector(
            weights=light_weights,
            device="mps",
            conf_threshold=0.5
        )

    def draw_tracked_vehicles(self, frame, tracked_vehicles):
        for obj in tracked_vehicles:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = int(obj["track_id"])
            class_name = obj["class_name"]
            foot = obj["foot"]

            label = f"{class_name} | ID {track_id}"


            color = (
                50 + (track_id * 40) % 180,
                50 + (track_id * 70) % 180,
                50 + (track_id * 90) % 180
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

            text_x = x1
            text_y = y1 - 6 if y1 > 20 else y1 + th + 6

            cv2.rectangle(
                frame,
                (text_x, text_y - th - 4),
                (text_x + tw + 4, text_y + 2),
                color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (text_x + 2, text_y),
                font,
                font_scale,
                (255, 255, 255),
                1
            )

            cv2.circle(frame, foot, 3, color, -1)

    def draw_traffic_lights(self, frame, light_detections):
        for obj in light_detections:
            x1, y1, x2, y2 = obj["bbox"]
            class_name = obj["class_name"]
            conf = obj["conf"]

            label = f"{class_name} {conf:.2f}"

            if "red" in class_name.lower():
                color = (0, 0, 255)
            elif "green" in class_name.lower():
                color = (0, 255, 0)
            elif "yellow" in class_name.lower():
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8 if y1 > 20 else y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1
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

            self.frame_idx += 1
            frame = cv2.resize(frame, (960, 540))

            tracked_vehicles = self.vehicle_detector.track(frame)

            if self.frame_idx % 5 == 0:
                self.last_light_detections = self.light_detector.process(frame)

            light_detections = self.last_light_detections

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
        light_weights="/Users/Documents/DL-Project/OD_DA2_NN_v1/config/best_yolov12_light.pt",
        video_path="/Users/Documents/DL-Project/OD_DA2_NN_v1/data/dataset/7572405326846.mp4"
    )
    engine.run()