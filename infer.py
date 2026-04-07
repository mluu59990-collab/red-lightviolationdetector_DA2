import cv2
from src.Vehicle_detect import VehicleDetector
from src.stop_line import StopLine
from src.violation_logic import StopLineViolationDetector
from src.light_state import TrafficLightStateManager
from src.visualizer import Visualizer


class InferEngine:
    def __init__(self, vehicle_weights, video_path):
        self.video_path = video_path
        self.frame_idx = 0

        self.vehicle_detector = VehicleDetector(
            weights=vehicle_weights,
            device="mps",
            conf_threshold=0.5,
            detect_classes={2, 3, 5, 7}
        )

        # Stop line mới của bạn
        self.stop_line = StopLine(
            pt1=(183, 355),
            pt2=(720, 352)
        )

        self.violation_detector = StopLineViolationDetector(self.stop_line)

        # ROI đèn giao thông
        # Bạn phải tự chỉnh lại cho khớp đúng video 960x540
        self.light_state = TrafficLightStateManager(
            roi = (632, 53, 799, 94),
            history_size=7,
            min_pixels=20
        )

        self.visualizer = Visualizer()

    def process_traffic_light(self, frame):
        self.light_state.update(frame)

    def process_vehicles(self, tracked_vehicles):
        red_light = self.light_state.is_red()

        for obj in tracked_vehicles:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = int(obj["track_id"])

            foot, is_violation = self.violation_detector.update(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                red_light=red_light
            )

            obj["foot"] = foot
            obj["is_violation"] = is_violation

        return tracked_vehicles

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

            self.process_traffic_light(frame)
            tracked_vehicles = self.process_vehicles(tracked_vehicles)

            self.stop_line.draw(frame)
            self.visualizer.draw_tracked_vehicles(frame, tracked_vehicles)

            self.visualizer.draw_hud(
                frame,
                self.light_state.is_red(),
                self.violation_detector.get_total_violations()
            )

            self.light_state.draw_debug(frame)

            cv2.imshow("Infer", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = InferEngine(
        vehicle_weights="/Users/Documents/DL-Project/OD_DA2_NN_v1/config/yolov12_best.pt",
        video_path="/Users/Documents/DL-Project/OD_DA2_NN_v1/data/dataset/7572405326846.mp4"
    )
    engine.run()