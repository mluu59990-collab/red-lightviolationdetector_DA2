from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(self, max_age=10):
        self.tracker = DeepSort(max_age=max_age)

    def process(self, frame, detections):
        track_inputs = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            class_id = det["class_id"]

            track_inputs.append((
                [x1, y1, x2 - x1, y2 - y1],
                conf,
                class_id
            ))

        tracks = self.tracker.update_tracks(track_inputs, frame=frame)

        outputs = []
        for t in tracks:
            if not t.is_confirmed():
                continue

            class_id = t.get_det_class()
            if class_id is None:
                continue

            x1, y1, x2, y2 = map(int, t.to_ltrb())
            track_id = t.track_id
            cx = (x1 + x2) // 2
            foot = (cx, y2)

            outputs.append({
                "track_id": track_id,
                "class_id": class_id,
                "bbox": (x1, y1, x2, y2),
                "foot": foot
            })

        return outputs