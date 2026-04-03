import cv2


class Visualizer:
    def draw_tracked_vehicles(self, frame, tracked_vehicles):
        for obj in tracked_vehicles:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = int(obj["track_id"])
            class_name = obj["class_name"]
            foot = obj["foot"]
            is_violation = obj.get("is_violation", False)

            if is_violation:
                color = (0, 0, 255)
                label = f"{class_name} | ID {track_id} | VIOLATION"
            else:
                color = (
                    50 + (track_id * 40) % 180,
                    50 + (track_id * 70) % 180,
                    50 + (track_id * 90) % 180
                )
                label = f"{class_name} | ID {track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

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

            cv2.circle(frame, foot, 4, (255, 0, 255), -1)

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

    def draw_hud(self, frame, is_red, total_violations):
        light_text = "RED" if is_red else "NOT RED"
        light_color = (0, 0, 255) if is_red else (0, 255, 0)

        cv2.putText(
            frame,
            f"LIGHT: {light_text}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            light_color,
            2
        )

        cv2.putText(
            frame,
            f"Violations: {total_violations}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )