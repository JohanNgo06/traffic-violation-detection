from ultralytics import YOLO

if __name__ == '__main__':
    # Tải mô hình gốc bản nano (nhẹ và nhanh nhất để test luồng)
    model = YOLO("yolo11n.pt") 
    
    model.train(
        data="dataset/data.yaml",
        epochs=30,       # Train thử 30 vòng
        imgsz=640,
        batch=8,         # Mức an toàn tránh tràn VRAM trên RTX 3050
        device=0,        # Ép chạy bằng GPU
        workers=4
    )