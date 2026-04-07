import math


class StopLineViolationDetector:
    def __init__(self, stop_line, min_movement=2):
        self.stop_line = stop_line
        self.last_points = {}
        self.violated_ids = set()
        self.min_movement = min_movement

    def get_footpoint(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int(y2))

    def point_line_side(self, point):
        """
        Trả về giá trị dấu của điểm so với đường thẳng stop line.
        > 0: một phía
        < 0: phía còn lại
        = 0: nằm đúng trên line
        """
        x, y = point
        x1, y1 = self.stop_line.pt1
        x2, y2 = self.stop_line.pt2

        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    def has_crossed_line(self, prev_point, curr_point):
        prev_side = self.point_line_side(prev_point)
        curr_side = self.point_line_side(curr_point)

        # Nếu 1 điểm ở một phía, điểm còn lại ở phía kia
        return prev_side * curr_side < 0

    def is_moving_enough(self, prev_point, curr_point):
        dx = curr_point[0] - prev_point[0]
        dy = curr_point[1] - prev_point[1]
        dist = math.sqrt(dx * dx + dy * dy)
        return dist >= self.min_movement

    def update(self, track_id, bbox, red_light=True):
        foot = self.get_footpoint(bbox)
        prev_foot = self.last_points.get(track_id)

        if prev_foot is not None and track_id not in self.violated_ids:
            if red_light and self.is_moving_enough(prev_foot, foot):
                if self.has_crossed_line(prev_foot, foot):
                    self.violated_ids.add(track_id)

        self.last_points[track_id] = foot
        return foot, (track_id in self.violated_ids)

    def get_total_violations(self):
        return len(self.violated_ids)