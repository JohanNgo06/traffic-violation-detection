# Hệ thống Phát hiện Hành vi Vi phạm Giao thông (YOLOv11 + MERN Stack)

## 1. Phân chia nhánh Git (5 Thành viên: 3 AI – 2 Web)
- `ai-model` (AI 1): Dữ liệu Roboflow & Huấn luyện mô hình (`best.pt`).
- `ai-logic` (AI 2): Tracking (ByteTrack/Supervision) & Thuật toán hình học bắt 3 lỗi.
- `ai-api` (AI 3): FastAPI Server (cổng `8000`), phát luồng MJPEG & gọi Webhook sang Node.js.
- `web-backend` (Web 1): Node.js + Express + MongoDB + Socket.io (cổng `5000`), xác thực JWT.
- `web-frontend` (Web 2): React + Vite + Tailwind CSS (cổng `5173`), HTML5 Canvas vẽ vạch kẻ ROI.

---

## 2. Quy ước Hệ thống & Cổng chạy (Ports)
- **Frontend (React):** `http://localhost:5173`
- **Backend (Node.js):** `http://localhost:5000`
- **AI Worker (FastAPI):** `http://localhost:8000`
- **MongoDB Local:** `mongodb://127.0.0.1:27017/traffic_violation_db`

### Cấu trúc thư mục dùng chung
```text
traffic-violation-detection/
├── client/          # React Frontend
├── server/          # Node.js Backend
├── ai_worker/       # Python FastAPI & YOLO
└── uploads/         # Lưu trữ file media dùng chung giữa Node.js và Python
    ├── videos/      # Video .mp4 tải lên
    ├── frames/      # Ảnh frame đầu tiên làm nền vẽ Canvas
    └── evidence/    # Ảnh bằng chứng vi phạm (full.jpg & crop.jpg)
```

---

## 3. Hợp đồng Dữ liệu Chuẩn (Bắt buộc dùng chung)

### Mã định danh cố định
- **`vehicleType`:** `"motorbike"` | `"car"` | `"truck"` | `"bus"`
- **`violationType`:**
  - `"RED_LIGHT"` (Vượt đèn đỏ)
  - `"WRONG_LANE"` (Đi sai làn)
  - `"WRONG_WAY"` (Đi ngược chiều)

### Cấu trúc JSON `roiConfig` (Tọa độ quy đổi theo kích thước pixel gốc của video)
```json
{
  "stopLine": [[200, 600], [1000, 600]],
  "redLightIntervals": [
    { "startSec": 10, "endSec": 35 }
  ],
  "lanes": [
    {
      "laneName": "Lan_Xe_May",
      "allowedVehicles": ["motorbike"],
      "polygon": [[200, 600], [500, 600], [450, 1000], [100, 1000]],
      "directionVector": [[300, 900], [320, 650]]
    }
  ]
}
```

### Danh sách API Endpoints (`http://localhost:5000/api`)
| Luồng | Method & Endpoint | Dữ liệu gửi (Request) | Dữ liệu trả về (Response) |
| :--- | :--- | :--- | :--- |
| React -> Node | `POST /auth/register` | `{ fullName, email, password }` | `{ message }` |
| React -> Node | `POST /auth/login` | `{ email, password }` | `{ token, user: { id, fullName, email } }` |
| React -> Node | `POST /sessions/upload` | `FormData`: `video` (`.mp4`), `title` | `{ session }` (`_id`, `videoUrl`, `firstFrameUrl`, `width`, `height`) |
| React -> Node | `PUT /sessions/:id/config` | `{ roiConfig }` | `{ session }` |
| React -> Node | `GET /sessions` | Header `Authorization` | Danh sách `VideoSession` của user |
| React -> Node | `DELETE /sessions/:id` | Header `Authorization` | `{ message }` |
| React -> Node | `POST /sessions/:id/start` | Header `Authorization` | `{ streamUrl: "http://localhost:8000/stream/:id" }` |
| React -> Node | `POST /sessions/:id/stop` | Header `Authorization` | `{ message }` |
| Python -> Node | `POST /violations` | `{ sessionId, trackId, vehicleType, violationType, timestampInVideo, fullImageUrl, cropImageUrl }` | `{ success: true }` (Node tự gắn `userId` & bắn Socket `'new_violation'`) |
| React -> Node | `GET /violations` | Query `?sessionId=&vehicleType=&violationType=` | Danh sách `Violation` của user |