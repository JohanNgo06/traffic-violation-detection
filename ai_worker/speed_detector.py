"""Track vehicles and overlay an approximate speed on an input video."""

import argparse
import math
import unicodedata
from collections import deque
from pathlib import Path
from statistics import median

import cv2
from ultralytics import YOLO


AI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AI_DIR.parent
DEFAULT_MODEL = AI_DIR / "weights" / "best.pt"
DEFAULT_VIDEO = PROJECT_DIR / "uploads" / "videos" / "video_test.mp4"
OUTPUT_DIR = PROJECT_DIR / "runs" / "ai2_speed_auto"

# Approximate heights in meters. For two-wheelers the box includes the rider.
VEHICLE_HEIGHTS = {
    "Xe may": 1.5,
    "O to": 1.55,
    "Xe tai": 3.0,
    "Xe buyt": 3.2,
    "Xe dap": 1.6,
}
VEHICLE_NAMES = {
    "Xe may": ("motorcycle", "motorcycles", "motorbike", "motorbikes", "scooter", "xemay"),
    "O to": ("car", "cars", "automobile", "oto", "xeoto"),
    "Xe tai": ("truck", "trucks", "lorry", "xetai"),
    "Xe buyt": ("bus", "buses", "xebus", "xebuyt"),
    "Xe dap": ("bicycle", "bicycles", "bike", "xedap"),
}
NAME_BY_CLASS = {alias: name for name, aliases in VEHICLE_NAMES.items() for alias in aliases}


def vehicle_name(label):
    text = unicodedata.normalize("NFKD", str(label).casefold().replace("đ", "d"))
    key = "".join(char for char in text if char.isascii() and char.isalnum())
    return NAME_BY_CLASS.get(key)


class SpeedEstimator:
    def __init__(self, fps):
        self.fps = fps
        self.minimum_frames = max(1, math.ceil(0.2 * fps))
        self.window_frames = max(self.minimum_frames, round(0.5 * fps))
        self.histories = {}

    def observe(self, track_id, name, bbox, frame_idx, frame_height):
        x1, y1, x2, y2 = bbox
        pixel_height = y2 - y1
        if pixel_height <= 0 or y1 <= 0 or y2 >= frame_height - 1:
            self.histories.pop(track_id, None)
            return None

        history = self.histories.get(track_id)
        if not history or history[-1][0] != frame_idx - 1:
            history = deque(maxlen=self.window_frames + 1)
            self.histories[track_id] = history
        meters_per_pixel = VEHICLE_HEIGHTS[name] / pixel_height
        history.append((frame_idx, (x1 + x2) / 2, y2, meters_per_pixel))

        first_frame = history[0][0]
        if frame_idx - first_frame < self.minimum_frames:
            return None

        # Fit image velocity over recent frames to reduce box jitter.
        middle_frame = (first_frame + frame_idx) / 2
        time_variance = x_motion = y_motion = 0.0
        for sample_frame, x, y, _ in history:
            time_offset = sample_frame - middle_frame
            time_variance += time_offset * time_offset
            x_motion += time_offset * x
            y_motion += time_offset * y
        pixels_per_second = math.hypot(x_motion, y_motion) * self.fps / time_variance
        local_scale = median(sample[3] for sample in history)
        return pixels_per_second * local_scale * 3.6

    def prune(self, visible_ids):
        for track_id in self.histories.keys() - visible_ids:
            del self.histories[track_id]


def draw_tag(frame, bbox, label, occupied, color):
    x1, y1, _, y2 = bbox
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.44, min(0.78, width / 2100))
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 1)
    if text_width > width - 8:
        scale *= max(1, width - 8) / text_width
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 1)
    tag_width, tag_height = text_width + 4, text_height + baseline + 4
    x = max(0, min(x1, width - tag_width))
    tops = (y1 - tag_height - 2, y1 + 2, y2 + 2, y2 - tag_height)
    tops = [max(0, min(top, height - tag_height)) for top in tops]

    def overlap(top):
        return sum(
            max(0, min(x + tag_width, right) - max(x, left))
            * max(0, min(top + tag_height, bottom) - max(top, other_top))
            for left, other_top, right, bottom in occupied
        )

    top = min(tops, key=overlap)
    occupied.append((x, top, x + tag_width, top + tag_height))
    origin = (x + 2, top + text_height + 2)
    cv2.putText(frame, label, origin, font, scale, color, 1, cv2.LINE_AA)


def annotate_frame(frame, result, estimator, frame_idx):
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        estimator.prune(set())
        return

    height, width = frame.shape[:2]
    ids = boxes.id.int().cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    coordinates = boxes.xyxy.cpu().tolist()
    visible_ids = set()
    tags = []
    color = (60, 230, 90)
    for track_id, class_id, bbox in zip(ids, classes, coordinates):
        name = vehicle_name(result.names[class_id])
        if name is None or not all(math.isfinite(value) for value in bbox):
            continue
        visible_ids.add(track_id)
        speed = estimator.observe(track_id, name, bbox, frame_idx, height)
        speed_text = "--" if speed is None else f"~{speed:.0f}"
        x1, y1, x2, y2 = map(round, bbox)
        visible_bbox = (max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2))
        cv2.rectangle(frame, visible_bbox[:2], visible_bbox[2:], color, 2, cv2.LINE_AA)
        tags.append((visible_bbox, f"{name} | ID {track_id} | {speed_text} km/h"))

    estimator.prune(visible_ids)
    occupied = []
    for bbox, label in tags:
        draw_tag(frame, bbox, label, occupied, color)


def run_video(video_path, model_path, device, seconds):
    output = OUTPUT_DIR / f"{video_path.stem}_speed.mp4"
    if output.resolve() == video_path.resolve():
        raise ValueError("Hay dung video goc tu thu muc uploads/videos.")

    capture = cv2.VideoCapture(str(video_path))
    writer = None
    frame_idx = 0
    try:
        ok, frame = capture.read()
        if not ok:
            raise OSError(f"Khong doc duoc video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        estimator = SpeedEstimator(fps)
        model = YOLO(str(model_path))
        height, width = frame.shape[:2]
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"Khong ghi duoc video: {output}")

        print("Dang uoc tinh toc do xe...", flush=True)
        while ok and (seconds is None or frame_idx / fps < seconds):
            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml", device=device, verbose=False
            )[0]
            annotate_frame(frame, result, estimator, frame_idx)
            writer.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Da xu ly {frame_idx} khung hinh.", flush=True)
            ok, frame = capture.read()
    except KeyboardInterrupt:
        print("\nDa dung xu ly.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frame_idx:
        print(f"Da luu video: {output}")


def main():
    parser = argparse.ArgumentParser(description="Uoc tinh toc do xe tu video bang YOLO + ByteTrack.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Duong dan video goc.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Duong dan file best.pt.")
    parser.add_argument("--device", default="cpu", help="Thiet bi nhan dien, vi du cpu hoac 0.")
    parser.add_argument("--seconds", type=float, help="So giay dau cua video de chay thu.")
    args = parser.parse_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds phai lon hon 0.")
    for path in (args.video, args.model):
        if not path.is_file():
            parser.error(f"Khong tim thay file: {path.resolve()}")
    try:
        run_video(args.video, args.model, args.device, args.seconds)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
