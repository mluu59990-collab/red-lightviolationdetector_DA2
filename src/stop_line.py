import cv2


class StopLine:
    def __init__(self, pt1, pt2, color=(0, 0, 255), thickness=3, label="STOP LINE"):
        self.pt1 = pt1
        self.pt2 = pt2
        self.color = color
        self.thickness = thickness
        self.label = label

    def draw(self, frame):
        cv2.line(frame, self.pt1, self.pt2, self.color, self.thickness, cv2.LINE_AA)
        cv2.putText(
            frame,
            self.label,
            (self.pt1[0], self.pt1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.color,
            2
        )
        return frame