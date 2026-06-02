import os
import sys
import math
import urllib.request
import cv2
import numpy as np

from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import image as image_lib

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"


def download_model(path: str) -> None:
    if os.path.exists(path):
        return
    print("Descargando modelo de hand landmarker...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("Modelo descargado:", path)


def select_video() -> str:
    candidate = sys.argv[1] if len(sys.argv) > 1 else "cat.mp4"
    if os.path.exists(candidate):
        return candidate
    return ""


def es_puno(hand_landmarks) -> bool:
    puntas = [8, 12, 16, 20]
    nudillos = [6, 10, 14, 18]
    for punta, nudillo in zip(puntas, nudillos):
        if hand_landmarks[punta].y < hand_landmarks[nudillo].y:
            return False
    return True


def es_palma_abierta(hand_landmarks) -> bool:
    puntas = [8, 12, 16, 20]
    bases = [5, 9, 13, 17]
    for punta, base in zip(puntas, bases):
        if hand_landmarks[punta].y > hand_landmarks[base].y:
            return False
    return True


def main() -> None:
    download_model(MODEL_FILENAME)

    video_path = select_video()
    if not video_path:
        print("No se encontró ningún video. Modo cámara solamente activado.")
    else:
        print(f"Usando video: {video_path}")

    hand_landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=MODEL_FILENAME),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")

    cap_video = None
    video_playing = False
    video_window_open = False

    if video_path:
        cap_video = cv2.VideoCapture(video_path)
        if cap_video.isOpened():
            video_playing = True
            print("Gatito encendido!")
        else:
            print("Error al abrir el video")
            cap_video = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = image_lib.Image(image_lib.ImageFormat.SRGB, rgb_frame)
            results = hand_landmarker.detect(mp_image)

            hay_puno = False
            hay_palma = False

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    mp_vision.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    )
                    if es_puno(hand_landmarks):
                        hay_puno = True
                    if es_palma_abierta(hand_landmarks):
                        hay_palma = True

            if hay_puno and not video_playing and video_path:
                cap_video = cv2.VideoCapture(video_path)
                if cap_video.isOpened():
                    video_playing = True
                    print("Gatito encendido!")
                else:
                    print("Error al abrir el video")
                    cap_video = None
            elif hay_puno and not video_playing and not video_path:
                print("No hay video disponible. Solo cámara activada.")

            if video_playing and hay_palma:
                if cap_video is not None:
                    cap_video.release()
                    cap_video = None
                video_playing = False
                if video_window_open:
                    cv2.destroyWindow("Gatito")
                    video_window_open = False
                print("Gatito apagado!")

            cv2.imshow("Scuba Cat", frame)

            if video_playing and cap_video is not None:
                ret_video, video_frame = cap_video.read()
                if not ret_video:
                    cap_video.release()
                    cap_video = None
                    video_playing = False
                    if video_window_open:
                        cv2.destroyWindow("Gatito")
                        video_window_open = False
                    print("Fin del video o error de lectura")
                else:
                    cv2.imshow("Gatito", video_frame)
                    video_window_open = True

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if cap_video is not None:
            cap_video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
