from ultralytics import YOLO

# Nạp mô hình vừa train xong
model = YOLO("runs/detect/train/weights/best.pt")

# Chạy nhận diện + theo dõi quỹ đạo từng xe
model.track(
    source="video_test.mp4",  # Tên file video của bạn
    show=True,                # Mở cửa sổ hiển thị video trực quan ngay lập tức
    save=True,                # Lưu video kết quả vào runs/detect/track
    conf=0.4,                 # Chỉ hiện những xe có độ tin cậy >= 40%
    tracker="bytetrack.yaml"
)