import bettercam
import cv2
import pyautogui
import threading
from ultralytics import YOLO
import time
import ctypes
import torch

# Определяем модель и проверяем наличие GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('bestfortnite.pt').to(device)

start_time = time.perf_counter()
box_constant = 1000
screensize = {'2560': ctypes.windll.user32.GetSystemMetrics(0), '1440': ctypes.windll.user32.GetSystemMetrics(1)}
screen_res_X = screensize['2560']  # Horizontal
screen_res_Y = screensize['1440']  # Vertical
print(f"Screen Resolution: {screen_res_X, screen_res_Y}")
screen_x = int(screen_res_X / 2)
screen_y = int(screen_res_Y / 2)
left, top = (screen_res_X - box_constant) // 2, (screen_res_Y - box_constant) // 2
right, bottom = left + box_constant, top + box_constant
detection_box = (left, top, right, bottom)
print(detection_box)


def handle_key(key):
    pyautogui.keyDown(key)
    pyautogui.keyUp(key)
    print(f"Нажата клавиша: {key}")


keys = ['d', 'f', 'j', 'k']
rectangles = [(150, 290), (310, 450), (470, 610), (630, 770)]

camera = bettercam.create(output_color="BGR", region=detection_box, max_buffer_len=10)


def process_frame(frame):
    start_time = time.perf_counter()
    predictions = model.predict(frame)
    end_time = time.perf_counter()
    frame_processing_time = end_time - start_time
    fps = 1 / frame_processing_time
    fps_int = int(fps)
    cv2.putText(frame, f"FPS: {fps_int}", (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (113, 116, 244), 2)
    return frame, predictions


while True:
    frame = camera.grab()
    if frame is not None:
        frame, predictions = process_frame(frame)

        # Определение прямоугольников
        rectangles = [(150, 700, 290, 855),
                      (310, 700, 450, 855),
                      (470, 700, 610, 855),
                      (630, 700, 770, 855)]

        for rect in rectangles:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (100, 255, 255), 2)

        for prediction in predictions:
            if prediction.boxes.xyxy.numel() > 0:
                for box in prediction.boxes.xyxy:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for i, (left, top, right, bottom) in enumerate(rectangles):
                        if (left <= x2 - 10 <= right) and (700 <= y2 <= 855):
                            thread = threading.Thread(target=handle_key, args=(keys[i],))
                            thread.start()
            else:
                print("Отсутствуют.")

        cv2.imshow('Vision', frame)
        #model.predict(frame, show=True)

    time.sleep(0.01)  # Минимальная задержка между кадрами

    if cv2.waitKey(1) & 0xFF == ord('0'):
        break

cv2.destroyAllWindows()

