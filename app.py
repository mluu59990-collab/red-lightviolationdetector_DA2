import cv2

video_path = "/Users/Documents/DL-Project/OD_DA2_NN_v1/data/dataset/7572405326846.mp4"
cap = cv2.VideoCapture(video_path)

paused = True
current_frame = None

def mouse_callback(event, x, y, flags, param):
    global current_frame
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        print(f"x={x}, y={y}")
        cv2.circle(current_frame, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(current_frame, f"({x},{y})", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.namedWindow("Infer")
cv2.setMouseCallback("Infer", mouse_callback)

while True:
    if not paused or current_frame is None:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = cv2.resize(frame, (960, 540))

    cv2.imshow("Infer", current_frame)
    key = cv2.waitKey(30) & 0xFF

    if key == 27:   # ESC
        break
    elif key == ord(" "):   # Space để pause/unpause
        paused = not paused

cap.release()
cv2.destroyAllWindows()