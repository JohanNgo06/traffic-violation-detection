from ultralytics import YOLO

if __name__ == '__main__':
    # Nạp mô hình v1 vừa train xong
    model = YOLO("ai_worker/weights/best.pt")

    # Chạy nhận diện + tracking trên video test và lưu video kết quả
    model.track(
        source="uploads/videos/video_test.mp4",
        conf=0.4,               # Ngưỡng tự tin 40%
        iou=0.5,
        show=True,              # Hiển thị cửa sổ xem trực tiếp
        save=True,              # Lưu video kết quả vào runs/detect/track
        device=0
    )