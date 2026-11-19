from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ai_worker/yolo11n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=40,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        name="v1"
    )