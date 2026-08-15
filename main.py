import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

img_peace = cv2.imread('peace.png')

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

HANDS_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

def place_image_direct(backround, overlay, x, y, size=(150, 150)):
    if overlay is None:
        return backround

    overlay_resized = cv2.resize(overlay, size)
    h, w, _ = overlay_resized.shape

    y1, y2 = y, y + h
    x1, x2 = x, x + w

    if y2 > backround.shape[0] or x2 > backround.shape[1] or x1 < 0 or y1 < 0:
        return backround

    backround[y1:y2, x1:x2] = overlay_resized
    return backround

def check_open_finger(landmarks):
    tips_ids = [8, 12, 16, 20]
    pips_ids = [7, 11, 15, 19]

    status = []

    for tip, pip in zip(tips_ids, pips_ids):
        if landmarks[tip].y < landmarks[pip].y:
            status.append(1)
        else:
            status.append(0)

    return status

def draw_landmarks_custom(frame, detection_result):
    if not detection_result.hand_landmarks:
        return frame

    h, w, _ = frame.shape

    for hand_landmarks in detection_result.hand_landmarks:
        pixel_points = []
        for landmark in hand_landmarks:
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            pixel_points.append((px, py))

            cv2.circle(frame, (px, py), 5, (0, 255, 0), cv2.FILLED)

        for connection in HANDS_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            cv2.line(frame, pixel_points[start_idx], pixel_points[end_idx], (255, 0, 0,), 2)

        finger_status = check_open_finger(hand_landmarks)

        wrist_x, wrist_y = pixel_points[0]

        if finger_status == [1, 1, 0, 0]:
            frame = place_image_direct(frame, img_peace, 20, 20)

    return frame


cap = cv2.VideoCapture(0)

with vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print('Kamera Tidak Ditemukan')
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        timestamp_ms = int(time.time() * 1000)

        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        frame = draw_landmarks_custom(frame, detection_result)

        cv2.imshow("Finger Tracker", frame)

        if cv2.waitKey(2) & 0xff == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()