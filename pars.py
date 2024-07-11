from ultralytics import YOLO
import locale

# Set preferred encoding
locale.getpreferredencoding = lambda: "UTF-8"

def train_model():
    # Load YOLO model
    model = YOLO("yolov8n.pt")
    # Train the model
    results = model.train(data="C:/Users/vovaf/Desktop/ru/data.yaml", epochs=1000, imgsz=640, batch=40, workers=8, dropout = 0.1, patience=0) #patience = 50
    return results

def export_model_to_onnx(model_path):
    # Загружаем модель YOLO
    model = YOLO(model_path)
  # Экспортируем модель в формат ONNX
    model.export(format="onnx")

if __name__ == "__main__":
    # Указываем путь к модели
    #train_model()
    model_path = "best.pt"
    export_model_to_onnx(model_path)
