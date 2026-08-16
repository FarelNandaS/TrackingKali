import sys
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rich.console import Console
from rich.text import Text
from rich.panel import Panel

console = Console()

img_peace = cv2.imread('img/peace.png')
img_angry = cv2.imread('img/angry.png')

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

LOGO_ART = r"""[bold green]
▀█▀ █▀█ ▄▀█ █▄▀ █ █▄ █ █▀▀ █▄▀ ▄▀█ █   █
 █  █▀▄ █▀█ █ █ █ █ ▀█ █▄█ █ █ █▀█ █▄▄ █
[/bold green]"""

HANDS_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (2, 5), (5, 6), (6, 7), (7, 8),
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

def check_open_finger(landmarks, hand_label):
    status = []

    tips_ids = [8, 12, 16, 20]
    pips_ids = [7, 11, 15, 19]

    if hand_label:
        if hand_label == 'Right':
            if landmarks[4].x >= landmarks[2].x:
                status.append(1)
            else:
                status.append(0)
        else:
            if landmarks[4].x <= landmarks[2].x:
                status.append(1)
            else:
                status.append(0)

    for tip, pip in zip(tips_ids, pips_ids):
        if landmarks[tip].y <= landmarks[pip].y:
            status.append(1)
        else:
            status.append(0)

    return status

def mode_besic_image(frame, detection_result):
    if not detection_result.hand_landmarks:
        return frame

    h, w, _= frame.shape

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

    return frame

def mode_gesture_image(frame, detection_result):
    frame = mode_besic_image(frame, detection_result)

    if not detection_result.hand_landmarks:
        return frame

    h, w, _ = frame.shape

    for hand_landmarks in detection_result.hand_landmarks:
        finger_status = check_open_finger(hand_landmarks)

        if finger_status == [1, 1, 0, 0]:
            frame = place_image_direct(frame, img_peace, 20, 20)
        elif finger_status == [0, 0, 0, 0]:
            frame = place_image_direct(frame, img_angry, 20, 20)

    return frame

def mode_count_finger(frame, detection_result):
    frame = mode_besic_image(frame, detection_result)

    if not detection_result.hand_landmarks:
        return frame

    total_finger_all = 0

    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            hand_label = "Right"
            if detection_result.handedness and idx < len(detection_result.handedness):
                hand_label = detection_result.handedness[idx][0].category_name

            finger_status = check_open_finger(hand_landmarks, hand_label)
            total_finger = finger_status.count(1)
            total_finger_all += total_finger

    h, w, _ = frame.shape

    cv2.putText(frame, str(total_finger_all), (40, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 4)

    return frame

def run_camera(mode):
    cap = cv2.VideoCapture(0)

    console.print("[bold yellow]Membuka Webcam... Tekan 'q' pada jendela kamera untuk kembali ke menu.[/bold yellow]")

    with vision.HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print('[bold red]Kamera Tidak Ditemukan[/bold red]')
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_ms = int(time.time() * 1000)

            detection_result = detector.detect_for_video(mp_image, timestamp_ms)

            if mode == '1':
                frame = mode_besic_image(frame, detection_result)
            elif mode == '2':
                frame = mode_gesture_image(frame, detection_result)
            elif mode == '3':
                frame = mode_count_finger(frame, detection_result)

            cv2.imshow("Finger Tracker", frame)

            if cv2.waitKey(2) & 0xff == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def main_menu():
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')

            console.print(LOGO_ART)

            menu_text = (
                "[cyan]Pilih fitur yang ingin dijalankan:[/cyan] \n"
                "[cyan] [1] Basic Finger Tracking (Landmark jari saja)[/cyan] \n"
                "[cyan] [2] Gesture Detector (Tampilkan Gambar/Stiker)[/cyan] \n"
                "[cyan] [3] Finger Count (Menghitung jari)[/cyan] \n"
                "[cyan] [0] Keluar Dari Program[/cyan] \n"
            )

            console.print(Panel(menu_text, title="[bold green]🖐 TRACKING KALI SYSTEM 🖐[/bold green]", subtitle="[dim]Gunakan Angka untuk Memilih[/dim]", border_style="green", width=60))

            pilihan = console.input("[bold yellow]Masukan Pilihan Anda: [/bold yellow]").strip()

            if pilihan == '1':
                run_camera(mode='1')
            elif pilihan == '2':
                run_camera(mode='2')
            elif pilihan == '3':
                run_camera(mode='3')
            elif pilihan == '0':
                os.system('cls' if os.name == 'nt' else 'clear')
                console.print('\n[bold green]Terimakasih telah mengunakan program ini! Sampai Jumpa 🖐[/bold green]')
                sys.exit()
            else:
                os.system('cls' if os.name == 'nt' else 'clear')
                console.print('\n[bold yellow][!] Pilihan tidak valid. Silahkan masukan angka yang valid.[/bold yellow]')
                time.sleep(2)
    except KeyboardInterrupt:
        os.system('cls' if os.name == 'nt' else 'clear')
        console.print('\n[bold green]Terimakasih telah mengunakan program ini! Sampai Jumpa 🖐[/bold green]')
        sys.exit()

if __name__ == "__main__":
    main_menu()