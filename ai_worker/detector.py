import argparse
import json
import math
import re
from collections import deque
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from ultralytics import YOLO

if __package__:
    from .rules import (
        RED_LIGHT, SPEEDING, VEHICLE_TYPES, WRONG_LANE,
        RedLightRule, SpeedingRule, WrongLaneRule, normalize_vehicle_type,
    )
else:
    from rules import (
        RED_LIGHT, SPEEDING, VEHICLE_TYPES, WRONG_LANE,
        RedLightRule, SpeedingRule, WrongLaneRule, normalize_vehicle_type,
    )

AI_DIR = Path(__file__).resolve().parent
MODEL_PATH = AI_DIR / "weights" / "best.pt"
TEST_VIDEO = AI_DIR.parent / "uploads" / "videos" / "video_test.mp4"
RUNS_DIR = AI_DIR.parent / "runs"
EVIDENCE_DIR = AI_DIR.parent / "uploads" / "evidence"
VEHICLE_LABELS = {"motorbike": "Xe may", "car": "O to", "truck": "Xe tai", "bus": "Xe buyt"}


def video_fps(fps):
    return fps if fps and math.isfinite(fps) and fps > 0 else 30.0


def draw_id(frame, label, x1, y1):
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, 1
    )
    x = max(0, min(x1 + 3, width - text_width - 3))
    y = min(max(y1 + text_height + 4, text_height + 2), height - baseline - 2)
    cv2.putText(frame, label, (x, y), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_sample_text(frame, text, position, scale, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, scale, (15, 15, 15), 4)
    cv2.putText(frame, text, position, font, scale, color, 2)


def draw_lanes(frame, lanes, label_positions=None):
    canvas = frame.copy()
    if not lanes:
        return canvas

    fill = frame.copy()
    contours = []
    for name, polygon, allowed in lanes:
        color = (0, 215, 255)
        if allowed == {"car", "truck"}:
            color = (255, 180, 35)
        elif allowed == {"motorbike", "bus"}:
            color = (80, 230, 70)
        contour = np.rint(polygon).astype(np.int32)
        if label_positions is None:
            cv2.fillPoly(fill, [contour], color)
        else:
            cv2.fillConvexPoly(fill, contour, color)
            cv2.polylines(canvas, [contour], True, color, 2, cv2.LINE_AA)
        contours.append((name, contour, color))

    cv2.addWeighted(fill, 0.16, canvas, 0.84, 0, dst=canvas)
    for name, contour, color in contours:
        if label_positions is None:
            cv2.polylines(canvas, [contour], True, color, 2, cv2.LINE_AA)
            x, y = np.rint(contour.mean(axis=0)).astype(int)
            draw_id(canvas, name, int(x), int(y))
        else:
            x, y = label_positions[name]
            scale = max(0.42, min(1.1, frame.shape[1] / 1450))
            draw_sample_text(canvas, name, (round(x), round(y)), scale, color)
    return canvas


def vehicle_label(track_id, vehicle_type, lane_name):
    name = VEHICLE_LABELS.get(vehicle_type, "Khong ro loai")
    return f"{name} | ID {track_id} | {lane_name or 'Lan -'}"


def draw_vehicle_tag(frame, label, bbox, occupied, text_color=(255, 255, 255), *, background=False):
    x1, y1, _, y2 = bbox
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.50, min(0.95, width / 1680))
    padding_x, padding_y = 6, 4
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 2)
    available_width = max(1, width - 2 * padding_x)
    if text_width > available_width:
        scale *= max(1, available_width - 2) / text_width
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 2)
    tag_width = text_width + 2 * padding_x
    tag_height = text_height + baseline + 2 * padding_y
    x = max(0, min(x1, width - tag_width))
    tops = (y1 - tag_height - 3, y1 + 3, y2 + 3, y2 - tag_height - 3)
    candidates = [max(0, min(top, height - tag_height)) for top in tops]

    def overlap(top):
        return sum(
            max(0, min(x + tag_width, right) - max(x, left))
            * max(0, min(top + tag_height, bottom) - max(top, other_top))
            for left, other_top, right, bottom in occupied
        )

    top = min(candidates, key=overlap)
    right, bottom = x + tag_width, top + tag_height
    occupied.append((x, top, right, bottom))
    origin = (x + padding_x, top + padding_y + text_height)
    if background:
        cv2.rectangle(frame, (x, top), (right - 1, bottom - 1), (15, 15, 15), cv2.FILLED)
    cv2.putText(frame, label, origin, font, scale, text_color, 2, cv2.LINE_AA)


class SpeedEstimator:
    def __init__(self, calibration):
        image_points = np.asarray(calibration["imagePoints"], dtype=np.float32)
        width = float(calibration["widthMeters"])
        length = float(calibration["lengthMeters"])
        if image_points.shape != (4, 2) or not np.isfinite(image_points).all():
            raise ValueError("Hieu chinh toc do can dung 4 diem anh hop le.")
        if not cv2.isContourConvex(image_points):
            raise ValueError("Bon diem do toc do phai tao thanh tu giac loi.")
        if not all(math.isfinite(value) and value > 0 for value in (width, length)):
            raise ValueError("Chieu rong va chieu dai do toc do phai lon hon 0.")

        # Same corner order in both views: far left, far right, near right, near left.
        road_points = np.array([(0, 0), (width, 0), (width, length), (0, length)],
                               dtype=np.float32)
        transform = cv2.getPerspectiveTransform(image_points, road_points)
        center = image_points.mean(axis=0, dtype=np.float64)
        center_scale = float(transform[2] @ (*center, 1.0))
        if (not np.isfinite(transform).all() or np.linalg.det(transform) == 0
                or not math.isfinite(center_scale) or abs(center_scale) < 1e-12):
            raise ValueError("Khong the quy doi toa do tu bon diem hieu chinh.")
        # Keep the projection scale positive on the visible side of the horizon.
        self.image_to_road = transform / center_scale
        self.history = {}

    def prune_missing_tracks(self, visible_ids):
        for track_id in self.history.keys() - visible_ids:
            del self.history[track_id]

    def observe(self, track_id, bbox, frame_idx, fps, frame_height):
        x1, _, x2, y2 = bbox
        road_x, road_y, scale = self.image_to_road @ ((x1 + x2) / 2, y2, 1.0)
        # Discard samples with a clipped ground contact or an invalid projection.
        if y2 >= frame_height - 1 or scale <= 1e-6:
            self.history.pop(track_id, None)
            return None

        point = (road_x / scale, road_y / scale)
        history = self.history.setdefault(track_id, deque())
        if history and history[-1][0] != frame_idx - 1:
            history.clear()
        history.append((frame_idx, point))

        # Show the first estimate after 0.2 seconds, then grow the moving window.
        window = max(1, round(0.5 * fps))
        while frame_idx - history[0][0] > window:
            history.popleft()
        first_frame, first_point = history[0]
        elapsed_frames = frame_idx - first_frame
        if elapsed_frames < max(1, math.ceil(0.2 * fps)):
            return None
        return math.dist(first_point, point) * fps / elapsed_frames * 3.6


class TrafficDetector:
    def __init__(self, session_id, roi_config, device="cpu", model_path=MODEL_PATH,
                 *, auto_speed=None, speed_limit_kmh=None):
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("session_id chi gom chu, so, dau gach ngang hoac gach duoi.")
        self.session_id = session_id
        self.device = device
        self.rule = WrongLaneRule(roi_config)
        self.red_light = None
        if "redLight" in roi_config:
            self.red_light = RedLightRule(roi_config["stopLine"], roi_config["redLight"])
        calibration = roi_config.get("speedCalibration")
        self.speed = None
        if auto_speed is None and calibration is not None:
            self.speed = SpeedEstimator(calibration)
        self.auto_speed = auto_speed
        self.speeding = None
        if speed_limit_kmh is not None:
            if self.auto_speed is None and self.speed is None:
                raise ValueError("Can bo uoc tinh toc do de kiem tra vuot toc do.")
            self.speeding = SpeedingRule(speed_limit_kmh)
        self.label_positions = None
        if self.rule.sample_style is not None:
            self.label_positions = {
                name: lane.get("labelPosition", polygon.mean(axis=0))
                for (name, polygon, _), lane in zip(self.rule.lanes, roi_config.get("lanes") or [])
            }
        self.model = YOLO(str(model_path))

    def process_frame(self, frame, frame_idx, fps):
        fps = video_fps(fps)
        result = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", device=self.device,
            save=False, show=False, verbose=False,
        )[0]
        canvas = draw_lanes(frame, self.rule.lanes, self.label_positions)
        occupied = []
        red_light = self.red_light
        speeding = self.speeding
        if red_light is not None:
            red_light.update_signal(frame, frame_idx, fps)
            state_colors = {"red": (0, 0, 255), "yellow": (0, 215, 255),
                            "green": (65, 200, 85), "unknown": (170, 170, 170)}
            color = state_colors[red_light.state]
            points = np.rint(red_light.points).astype(np.int32)
            thickness = max(3, round(frame.shape[1] / 340))
            cv2.polylines(canvas, [points], False, (20, 20, 20), thickness + 4, cv2.LINE_AA)
            cv2.polylines(canvas, [points], False, (0, 0, 255), thickness, cv2.LINE_AA)
            for (start_x, start_y), (end_x, end_y) in zip(points, points[1:]):
                occupied.append(
                    (min(start_x, end_x), min(start_y, end_y) - thickness,
                     max(start_x, end_x) + 1, max(start_y, end_y) + thickness + 1)
                )
            for name, (x1, y1, x2, y2) in red_light.lamps.items():
                border = color if red_light.cycle is None and name == red_light.state else (120, 120, 120)
                cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), border, 1)
            status = {"red": "DO", "yellow": "VANG", "green": "XANH",
                      "unknown": "CHUA RO"}
            label = f"DEN: {status[red_light.state]}"
            if red_light.cycle is not None:
                label += " | MO PHONG"
            draw_vehicle_tag(canvas, label, (8, 2, 8, 2),
                             occupied, color, background=True)
        annotated = canvas.copy()
        violations = []
        boxes = result.boxes
        track_ids, coordinates, class_ids = [], [], []
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
        visible_ids = set(track_ids)
        self.rule.prune_missing_tracks(visible_ids)
        if red_light is not None:
            red_light.prune_missing_tracks(visible_ids)
        if self.speed is not None:
            self.speed.prune_missing_tracks(visible_ids)
        if self.auto_speed is not None:
            self.auto_speed.prune(visible_ids)
        if speeding is not None:
            speeding.prune_missing_tracks(visible_ids)
        tags = []

        for track_id, bbox, class_id in zip(track_ids, coordinates, class_ids):
            vehicle_type = normalize_vehicle_type(result.names[class_id])
            pixel_bbox = tuple(round(value) for value in bbox)
            lane_name, wrong, active = self.rule.observe_frame(
                track_id, bbox, vehicle_type, frame_idx, fps, frame.shape
            )
            if active and not self.rule.has_violation(track_id):
                violation = self._save_evidence(
                    frame, canvas, pixel_bbox, track_id, vehicle_type, frame_idx / fps,
                )
                if violation is not None:
                    self.rule.mark_violation(track_id)
                    violations.append(violation)

            if red_light is not None and vehicle_type is not None:
                crossing = red_light.observe(track_id, bbox, frame_idx, fps)
                if crossing is not None:
                    violation = self._save_evidence(
                        frame, canvas, pixel_bbox, track_id, vehicle_type, crossing, RED_LIGHT,
                    )
                    if violation is not None:
                        red_light.mark_violation(track_id)
                        violations.append(violation)

            x1, y1, x2, y2 = pixel_bbox
            label = vehicle_label(track_id, vehicle_type, lane_name)
            if vehicle_type is None and (
                self.label_positions is not None or red_light is not None or self.auto_speed is not None
            ):
                continue
            if self.label_positions is not None:
                red = active
                color = (190, 190, 190) if lane_name is None or wrong else (65, 200, 85)
            else:
                red = self.rule.has_violation(track_id)
                color = (0, 255, 0)
                if not self.rule.lanes and (red_light is not None or self.auto_speed is not None):
                    label = f"{VEHICLE_LABELS[vehicle_type]} | ID {track_id}"
            kmh = None
            if vehicle_type is not None and (self.auto_speed is not None or self.speed is not None):
                if self.auto_speed is not None:
                    kmh = self.auto_speed.observe(
                        track_id, VEHICLE_LABELS[vehicle_type], bbox, frame_idx, frame.shape[0],
                    )
                    prefix = "~"
                else:
                    kmh = self.speed.observe(track_id, bbox, frame_idx, fps, frame.shape[0])
                    prefix = ""
                speed_text = "--"
                if kmh is not None and math.isfinite(kmh):
                    decimals = 1 if speeding is not None else 0
                    speed_text = f"{prefix}{kmh:.{decimals}f}"
                label += f" | {speed_text} km/h"
            if red:
                label += " | SAI LAN"
            if red_light is not None and track_id in red_light.violated_ids:
                red = True
                label += " | VUOT DEN DO"
            if speeding is not None and speeding.observe(track_id, kmh, frame_idx, fps):
                red = True
                label += " | VUOT TOC DO"
                if track_id not in speeding.violated_ids:
                    details = {
                        "speedKmh": kmh,
                        "speedLimitKmh": speeding.limit_kmh,
                        "speedSource": "automatic" if self.auto_speed is not None else "calibrated",
                    }
                    violation = self._save_evidence(
                        frame, canvas, pixel_bbox, track_id, vehicle_type, frame_idx / fps,
                        SPEEDING, speed_details=details,
                    )
                    if violation is not None:
                        speeding.mark_violation(track_id)
                        violations.append(violation)
            if red:
                color = (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3 if red else 2)
            tags.append((red, pixel_bbox, label, color))

        for _, bbox, label, text_color in sorted(tags, key=lambda tag: not tag[0]):
            draw_vehicle_tag(annotated, label, bbox, occupied, text_color)
        return annotated, violations

    def _save_evidence(self, original, canvas, bbox, track_id, vehicle_type, t_sec,
                       violation_type=WRONG_LANE, *, speed_details=None):
        x1, y1, x2, y2 = bbox
        height, width = original.shape[:2]
        crop = original[
            max(0, min(height, y1 - 20)):max(0, min(height, y2 + 20)),
            max(0, min(width, x1 - 20)):max(0, min(width, x2 + 20)),
        ]
        if not crop.size:
            return None

        full = canvas.copy()
        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"VIOLATION: {violation_type} (ID:{track_id})"
        if speed_details is not None:
            prefix = "~" if speed_details["speedSource"] == "automatic" else ""
            label += f" | {prefix}{speed_details['speedKmh']:.1f} > {speed_details['speedLimitKmh']:g} km/h"
        draw_id(full, label, x1, y1)
        prefix = f"{self.session_id}_{track_id}_{violation_type}"
        full_name = f"{prefix}_full.jpg"
        crop_name = f"{prefix}_crop.jpg"
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        for name, image in ((full_name, full), (crop_name, crop)):
            success, jpeg = cv2.imencode(".jpg", image)
            if not success:
                raise OSError(f"Khong ma hoa duoc anh: {name}")
            (EVIDENCE_DIR / name).write_bytes(jpeg.tobytes())

        event = {
            "sessionId": self.session_id,
            "trackId": track_id,
            "vehicleType": vehicle_type,
            "violationType": violation_type,
            "timestampInVideo": round(t_sec, 2),
            "fullImageUrl": f"/uploads/evidence/{full_name}",
            "cropImageUrl": f"/uploads/evidence/{crop_name}",
        }
        if speed_details is not None:
            event.update(speed_details)
        return event


def track_video(video_path, device="cpu", save=False):
    model = YOLO(str(MODEL_PATH))
    yield from model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        stream=True,
        device=device,
        save=save,
        project=str(RUNS_DIR),
        name="ai2_tracking",
        verbose=False,
    )


def draw_track_ids(result):
    frame = result.plot(labels=False, line_width=2)
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return frame, []

    track_ids = boxes.id.int().cpu().tolist()
    coordinates = boxes.xyxy.int().cpu().tolist()
    for (x1, y1, _, _), track_id in zip(coordinates, track_ids):
        draw_id(frame, f"ID {track_id}", x1, y1)
    return frame, track_ids


def create_writer(output, fps, frame):
    height, width = frame.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        writer.release()
        raise OSError(f"Khong ghi duoc video: {output}")
    return writer


def run_tracking_demo(video_path, device):
    if not video_path.is_file():
        raise FileNotFoundError(f"Khong tim thay video: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Khong mo duoc video: {video_path}")
        fps = video_fps(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()

    output = RUNS_DIR / "ai2_tracking_id" / f"{video_path.stem}_id.mp4"
    writer = None
    seen_ids = set()
    print("Dang xu ly video...", flush=True)
    try:
        for result in track_video(video_path, device=device):
            frame, track_ids = draw_track_ids(result)
            if writer is None:
                writer = create_writer(output, fps, frame)
            writer.write(frame)
            new_ids = set(track_ids) - seen_ids
            if new_ids:
                print("ID xe moi:", sorted(new_ids))
                seen_ids.update(new_ids)
    finally:
        if writer is not None:
            writer.release()

    if writer is None:
        raise OSError(f"Video khong co khung hinh doc duoc: {video_path}")
    print(f"Da luu video: {output}")
    print(f"Tong so ID: {len(seen_ids)}")


def pick_lane(frame, title="Bam 4 goc lan xe | Enter: Xac nhan | R: Chon lai | Esc: Thoat"):
    height, width = frame.shape[:2]
    scale = min(1.0, 1150 / width, 650 / height)
    preview = cv2.resize(frame, (round(width * scale), round(height * scale)))
    preview_height, preview_width = preview.shape[:2]
    points = []

    def add_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.imshow(title, preview)
    cv2.setMouseCallback(title, add_point)
    try:
        while True:
            view = preview.copy()
            for index, point in enumerate(points, start=1):
                cv2.circle(view, point, 5, (0, 215, 255), -1)
                draw_id(view, str(index), *point)
            if len(points) >= 2:
                cv2.polylines(view, [np.asarray(points, dtype=np.int32)], len(points) == 4,
                              (0, 215, 255), 2)
            cv2.imshow(title, view)
            key = cv2.waitKey(25) & 0xFF
            if key == 27:
                return None
            try:
                visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                return None
            if visible < 1:
                return None
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                if cv2.isContourConvex(np.asarray(points, dtype=np.int32)):
                    return [[round(x * width / preview_width), round(y * height / preview_height)]
                            for x, y in points]
                print("Hay bam 4 goc theo thu tu quanh vien vung. Nhan R de chon lai.")
    finally:
        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass


def load_roi_config(path, frame_shape):
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    reference = config.pop("referenceSize", None)
    if reference is None:
        return config

    ref_width, ref_height = reference["width"], reference["height"]
    if not all(math.isfinite(value) and value > 0 for value in (ref_width, ref_height)):
        raise ValueError("referenceSize phai co width va height lon hon 0.")
    height, width = frame_shape[:2]
    scale_x, scale_y = width / ref_width, height / ref_height
    for lane in config.get("lanes") or []:
        lane["polygon"] = [[x * scale_x, y * scale_y] for x, y in lane["polygon"]]
        if "labelPosition" in lane:
            x, y = lane["labelPosition"]
            lane["labelPosition"] = [x * scale_x, y * scale_y]
    if "sampleOverlay" in config:
        config["sampleOverlay"]["sideMargin"] *= scale_x
    if "speedCalibration" in config:
        calibration = config["speedCalibration"]
        calibration["imagePoints"] = [
            [x * scale_x, y * scale_y] for x, y in calibration["imagePoints"]
        ]
    if "redLight" in config:
        config["stopLine"] = [[x * scale_x, y * scale_y] for x, y in config["stopLine"]]
        settings = config["redLight"]
        settings["lamps"] = {
            name: [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            for name, (x1, y1, x2, y2) in settings["lamps"].items()
        }
        settings["lineMargin"] = settings.get("lineMargin", 3) * scale_y
    return config


def calibrate_speed(args):
    capture = cv2.VideoCapture(str(args.video))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise OSError(f"Khong doc duoc video: {args.video}")

    print("Bam theo thu tu: xa trai, xa phai, gan phai, gan trai.", flush=True)
    points = pick_lane(frame, "Hieu chinh toc do | Enter: Luu | R: Chon lai | Esc: Thoat")
    if points is None:
        return
    height, width = frame.shape[:2]
    config = {
        "referenceSize": {"width": width, "height": height},
        "speedCalibration": {
            "imagePoints": points,
            "widthMeters": args.road_width,
            "lengthMeters": args.road_length,
        },
    }
    SpeedEstimator(config["speedCalibration"])
    with args.calibrate_speed.open("x", encoding="utf-8") as output:
        json.dump(config, output, indent=2)
        output.write("\n")
    print(f"Da luu hieu chinh toc do: {args.calibrate_speed.resolve()}")


def run_wrong_lane_demo(args):
    capture = cv2.VideoCapture(str(args.video))
    writer = None
    try:
        ok, frame = capture.read()
        if not ok:
            raise OSError(f"Khong doc duoc video: {args.video}")
        fps = video_fps(capture.get(cv2.CAP_PROP_FPS))
        if args.pick_lane:
            polygon = pick_lane(frame)
            if polygon is None:
                return
            config = {"lanes": [{"laneName": "Lan demo", "polygon": polygon,
                                 "allowedVehicles": args.allowed}]}
        elif args.roi:
            config = load_roi_config(args.roi, frame.shape)
        else:
            config = {}
        speed_path = getattr(args, "speed_calibration", None)
        if speed_path is not None:
            speed_config = load_roi_config(speed_path, frame.shape).get("speedCalibration")
            if speed_config is None:
                raise ValueError(f"Khong tim thay speedCalibration trong {speed_path}.")
            config["speedCalibration"] = speed_config
        speed_limit_kmh = getattr(args, "speed_limit", None)
        speeding_enabled = speed_limit_kmh is not None
        auto_speed = None
        if getattr(args, "auto_speed", False) or (speeding_enabled and "speedCalibration" not in config):
            if __package__:
                from .speed_detector import SpeedEstimator as AutoSpeedEstimator
            else:
                from speed_detector import SpeedEstimator as AutoSpeedEstimator

            auto_speed = AutoSpeedEstimator(fps)
        lane_enabled = bool(config.get("lanes"))
        red_light_enabled = "redLight" in config
        speed_enabled = auto_speed is not None or "speedCalibration" in config
        if not (lane_enabled or red_light_enabled or speed_enabled):
            raise ValueError("Cau hinh can co lan duong, den giao thong hoac vung do toc do.")

        detector = TrafficDetector(f"demo_{uuid4().hex[:12]}", config, device=args.device,
                                   auto_speed=auto_speed, speed_limit_kmh=speed_limit_kmh)
        mode = "red_light" if red_light_enabled else "wrong_lane"
        if speeding_enabled:
            mode = f"{mode}_speeding" if red_light_enabled or lane_enabled else "speeding"
        elif speed_enabled:
            speed_mode = "speed_auto" if auto_speed is not None else "speed"
            mode = f"{mode}_{speed_mode}" if red_light_enabled or lane_enabled else speed_mode
        output = RUNS_DIR / f"ai2_{mode}" / f"{args.video.stem}_{mode}.mp4"
        writer = create_writer(output, fps, frame)
        frame_idx = 0
        total_violations = 0
        total_speeding = 0
        while ok and (args.seconds is None or frame_idx / fps < args.seconds):
            annotated, violations = detector.process_frame(frame, frame_idx, fps)
            writer.write(annotated)
            for violation in violations:
                message = (f"{violation['violationType']} | ID {violation['trackId']} | "
                           f"{violation['vehicleType']} | {violation['timestampInVideo']:.2f}s")
                if violation["violationType"] == SPEEDING:
                    total_speeding += 1
                    prefix = "~" if violation["speedSource"] == "automatic" else ""
                    message += (f" | {prefix}{violation['speedKmh']:.1f} km/h"
                                f" | Gioi han {violation['speedLimitKmh']:g} km/h")
                print(message, flush=True)
            total_violations += len(violations)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Da xu ly {frame_idx} khung hinh.", flush=True)
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if red_light_enabled or lane_enabled:
        summary = "Tong so loi" if red_light_enabled or speeding_enabled else "Tong so loi sai lan"
        print(f"{summary}: {total_violations}")
    if speeding_enabled:
        print(f"Tong so loi vuot toc do: {total_speeding}")
    print(f"Da luu video: {output}")


def main():
    parser = argparse.ArgumentParser(description="Theo doi xe va thu quy tac giao thong.")
    parser.add_argument("--video", type=Path, default=TEST_VIDEO)
    parser.add_argument("--device", default="cpu")
    lane_source = parser.add_mutually_exclusive_group()
    lane_source.add_argument("--pick-lane", action="store_true", help="Bam 4 goc lan tren anh.")
    lane_source.add_argument("--roi", type=Path, help="Duong dan file JSON theo cau truc roiConfig.")
    lane_source.add_argument("--calibrate-speed", type=Path,
                             help="Chon vung do toc do va luu vao file JSON moi.")
    speed_source = parser.add_mutually_exclusive_group()
    speed_source.add_argument("--speed-calibration", type=Path,
                              help="File JSON hieu chinh toc do cho ca khung hinh.")
    speed_source.add_argument("--auto-speed", action="store_true",
                              help="Uoc tinh toc do tu kich thuoc tham chieu cua xe.")
    parser.add_argument("--speed-limit", type=float, nargs="?", const=30.0,
                        help="Bat bao loi vuot toc do theo gioi han km/h (mac dinh 30).")
    parser.add_argument("--road-width", type=float, help="Chieu rong thuc cua vung do (met).")
    parser.add_argument("--road-length", type=float, help="Chieu dai thuc cua vung do (met).")
    parser.add_argument("--allowed", choices=VEHICLE_TYPES, nargs="+",
                        help="Loai xe duoc di trong lan chon bang --pick-lane.")
    parser.add_argument("--seconds", type=float, help="Gioi han so giay video de chay thu.")
    args = parser.parse_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds phai lon hon 0.")
    if args.speed_limit is not None and (not math.isfinite(args.speed_limit) or args.speed_limit <= 0):
        parser.error("--speed-limit phai lon hon 0.")
    if args.pick_lane and args.allowed is None:
        parser.error("--pick-lane can them --allowed de khai bao loai xe duoc phep.")
    if args.calibrate_speed:
        dimensions = (args.road_width, args.road_length)
        if not all(value is not None and math.isfinite(value) and value > 0
                   for value in dimensions):
            parser.error("--calibrate-speed can --road-width va --road-length lon hon 0.")
        if args.calibrate_speed.exists():
            parser.error(f"{args.calibrate_speed} da ton tai. Hay chon ten file moi.")
        calibrate_speed(args)
    elif args.pick_lane or args.roi or args.speed_calibration or args.auto_speed or args.speed_limit is not None:
        run_wrong_lane_demo(args)
    else:
        run_tracking_demo(args.video, args.device)


if __name__ == "__main__":
    main()
