import cv2
import numpy as np
from collections import deque, Counter


class TrafficLightStateManager:
    def __init__(self, roi, history_size=7, min_pixels=20):
        """
        roi: (x1, y1, x2, y2) vùng chứa đèn giao thông trên frame đã resize
        history_size: số frame dùng để vote trạng thái
        min_pixels: số pixel tối thiểu để coi là có màu hợp lệ
        """
        self.roi = roi
        self.history = deque(maxlen=history_size)
        self.current_state = "UNKNOWN"
        self.min_pixels = min_pixels

        self.last_scores = {
            "RED": 0,
            "YELLOW": 0,
            "GREEN": 0
        }

    def update(self, frame):
        x1, y1, x2, y2 = self.roi
        roi_img = frame[y1:y2, x1:x2]

        if roi_img.size == 0:
            self.current_state = "UNKNOWN"
            return

        state, scores = self._detect_state_from_roi(roi_img)
        self.last_scores = scores

        if state != "UNKNOWN":
            self.history.append(state)

        if len(self.history) > 0:
            counter = Counter(self.history)
            self.current_state = counter.most_common(1)[0][0]
        else:
            self.current_state = "UNKNOWN"

    def _detect_state_from_roi(self, roi_img):
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        h, w = hsv.shape[:2]
        if h < 3 or w < 3:
            return "UNKNOWN", {"RED": 0, "YELLOW": 0, "GREEN": 0}

        left = hsv[:, 0:w // 3]
        center = hsv[:, w // 3: 2 * w // 3]
        right = hsv[:, 2 * w // 3:w]

        # RED có 2 khoảng
        lower_red1 = np.array([0, 120, 120])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 120, 120])
        upper_red2 = np.array([180, 255, 255])

        # YELLOW
        lower_yellow = np.array([15, 120, 120])
        upper_yellow = np.array([40, 255, 255])

        # GREEN
        lower_green = np.array([40, 120, 120])
        upper_green = np.array([90, 255, 255])

        red_score = self._count_color(left, lower_red1, upper_red1, lower_red2, upper_red2)
        yellow_score = self._count_color(center, lower_yellow, upper_yellow)
        green_score = self._count_color(right, lower_green, upper_green)

        scores = {
            "RED": red_score,
            "YELLOW": yellow_score,
            "GREEN": green_score
        }

        best_state = max(scores, key=scores.get)
        best_score = scores[best_state]

        if best_score < self.min_pixels:
            return "UNKNOWN", scores

        return best_state, scores

    def _count_color(self, hsv_img, lower1, upper1, lower2=None, upper2=None):
        mask1 = cv2.inRange(hsv_img, lower1, upper1)

        if lower2 is not None and upper2 is not None:
            mask2 = cv2.inRange(hsv_img, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return cv2.countNonZero(mask)

    def is_red(self):
        return self.current_state == "RED"

    def get_state(self):
        return self.current_state

    def draw_debug(self, frame):
        x1, y1, x2, y2 = self.roi

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"LIGHT: {self.current_state}",
            (x1, y1 - 10 if y1 > 20 else y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        score_text = (
            f"R:{self.last_scores['RED']} "
            f"Y:{self.last_scores['YELLOW']} "
            f"G:{self.last_scores['GREEN']}"
        )

        score_y = y2 + 20 if y2 + 20 < frame.shape[0] else y2 - 10

        cv2.putText(
            frame,
            score_text,
            (x1, score_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )