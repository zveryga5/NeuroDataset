import sys
import win32api
import os
from PyQt5.QtWidgets import QWidget, QApplication, QDesktopWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QTimer, QRect, QElapsedTimer
from ultralytics import YOLO
import bettercam
import cv2
from datetime import datetime


class DetectionBox(QWidget):
    def __init__(self, model_path):
        super().__init__()
        # Устанавливаем свойства окна
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Устанавливаем размер окна в размер экрана
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, screen.width(), screen.height())

        self.crosshair_color = QColor(0, 255, 0)
        self.crosshair_thickness = 2

        self.coord = (0, 0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_label)
        self.timer.start(10)

        self.fps_timer = QElapsedTimer()
        self.frame_count = 0
        self.fps = 0

        # Загружаем модель YOLOv8
        self.model = YOLO(model_path)
        self.detections = []

        # Инициализируем захват экрана с помощью bettercam
        self.screensize = {'2560': screen.width(), '1440': screen.height()}
        self.screen_res_X = self.screensize['2560']  # Horizontal
        self.screen_res_Y = self.screensize['1440']  # Vertical
        self.box_constant = 1000
        self.left, self.top = (self.screen_res_X - self.box_constant) // 2, (self.screen_res_Y - self.box_constant) // 2
        self.right, self.bottom = self.left + self.box_constant, self.top + self.box_constant
        self.detection_box = (self.left, self.top, self.right, self.bottom)
        self.camera = bettercam.create(output_color="BGR", region=self.detection_box, max_buffer_len=10)

        self.screenshot_path = r"C:\Users\vovaf\Desktop\ru\image"
        self.annotation_path = r"C:\Users\vovaf\Desktop\ru\razm"
        os.makedirs(self.screenshot_path, exist_ok=True)
        os.makedirs(self.annotation_path, exist_ok=True)

    def update_label(self):
        self.coord = win32api.GetCursorPos()
        self.update()

    def detect_objects(self):
        frame = self.camera.grab()
        if frame is not None:
            # Преобразуем изображение в формат RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame)
            # Фильтруем детекции по уверенности
            self.detections = []
            self.confidences = []
            for i, conf in enumerate(results[0].boxes.conf):
                if conf > 0.85:  # Указываем процент уверенности -------------------------------------------------------------------------------------------------------------------
                    self.detections.append(results[0].boxes.xyxy[i])
                    self.confidences.append(conf)
            if self.detections:
                self.save_screenshot(frame)
            print(f"Detections: {self.detections}")

    def save_screenshot(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_base_name = f"image_{timestamp}"
        screenshot_file = os.path.join(self.screenshot_path, f"{file_base_name}.png")
        annotation_file = os.path.join(self.annotation_path, f"{file_base_name}.txt")

        # Сохраняем изображение
        cv2.imwrite(screenshot_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Сохраняем разметку в формате YOLO
        height, width, _ = frame.shape
        with open(annotation_file, 'w') as f:
            for i, box in enumerate(self.detections):
                x1, y1, x2, y2 = map(int, box)
                confidence = self.confidences[i]
                # Преобразуем координаты в формат YOLO
                x_center = (x1 + x2) / 2.0 / width
                y_center = (y1 + y2) / 2.0 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                f.write(f"0 {x_center:.8f} {y_center:.8f} {w:.8f} {h:.8f} {confidence:.2f}\n")

    def paintEvent(self, event):
        self.detect_objects()  # Обновляем детекции на каждом кадре

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.crosshair_color, self.crosshair_thickness, Qt.SolidLine)
        painter.setPen(pen)

        try:
            for i, box in enumerate(self.detections):
                x1, y1, x2, y2 = map(int, box)
                confidence = self.confidences[i] * 100  # Преобразуем уверенность в проценты
                rect = QRect(x1 + self.left, y1 + self.top, x2 - x1, y2 - y1)  # Смещаем координаты
                painter.drawRect(rect)
                painter.drawText(x1 + self.left, y1 + self.top - 10, f"объект ({confidence:.2f}%)")
        except Exception as e:
            print(f"Ошибка: {e}")

        self.update_fps()
        painter.setPen(QColor(255, 0, 0))
        painter.setFont(QFont("Arial", 12))
        painter.drawText(10, 30, f"FPS: {self.fps}")

    def update_fps(self):
        if not self.fps_timer.isValid():
            self.fps_timer.start()
        else:
            self.frame_count += 1
            elapsed = self.fps_timer.elapsed()
            if elapsed >= 1000:
                self.fps = self.frame_count * 1000 / elapsed
                self.frame_count = 0
                self.fps_timer.restart()


if __name__ == '__main__':
    model_path = "best.pt"  # YOLOv8

    if not os.path.exists(model_path):
        print(f"Файл модели не найден: {model_path}")
        sys.exit(1)

    app = QApplication(sys.argv)
    overlay = DetectionBox(model_path)
    overlay.show()
    app.exec()
