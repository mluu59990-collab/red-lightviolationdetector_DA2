class StopLineViolationDetector:
    def __init__(self, stop_line):
        self.stop_line = stop_line
        self.last_points = {}
        self.violated_ids = set()

    def get_footpoint(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int(y2))

    def orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-6:
            return 0
        return 1 if val > 0 else 2

    def on_segment(self, p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    def segments_intersect(self, p1, q1, p2, q2):
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and self.on_segment(p1, p2, q1):
            return True
        if o2 == 0 and self.on_segment(p1, q2, q1):
            return True
        if o3 == 0 and self.on_segment(p2, p1, q2):
            return True
        if o4 == 0 and self.on_segment(p2, q1, q2):
            return True

        return False

    def is_moving_forward(self, prev_foot, curr_foot):
        return curr_foot[1] > prev_foot[1]

    def update(self, track_id, bbox, red_light=True):
        foot = self.get_footpoint(bbox)
        prev_foot = self.last_points.get(track_id)

        if prev_foot is not None and track_id not in self.violated_ids:
            if red_light and self.is_moving_forward(prev_foot, foot):
                crossed = self.segments_intersect(
                    prev_foot, foot,
                    self.stop_line.pt1, self.stop_line.pt2
                )
                if crossed:
                    self.violated_ids.add(track_id)

        self.last_points[track_id] = foot
        return foot, (track_id in self.violated_ids)

    def get_total_violations(self):
        return len(self.violated_ids)