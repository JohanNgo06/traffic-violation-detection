import argparse
import json
import math
import re
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from ultralytics import YOLO

if __package__:
    from .rules import RED_LIGHT, VEHICLE_TYPES, WRONG_LANE, RedLightRule, WrongLaneRule, normalize_vehicle_type
else:
    from rules import RED_LIGHT, VEHICLE_TYPES, WRONG_LANE, RedLightRule, WrongLaneRule, normalize_vehicle_type

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
    else:
        cv2.putText(frame, label, origin, font, scale, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(frame, label, origin, font, scale, text_color, 2, cv2.LINE_AA)


class TrafficDetector:
    def __init__(self, session_id, roi_config, device="cpu", model_path=MODEL_PATH):
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("session_id chi gom chu, so, dau gach ngang hoac gach duoi.")
        self.session_id = session_id
        self.device = device
        self.rule = WrongLaneRule(roi_config)
        self.red_light = None
        if "redLight" in roi_config:
            self.red_light = RedLightRule(roi_config["stopLine"], roi_config["redLight"])
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
        if boxes is None or boxes.id is None:
            self.rule.prune_missing_tracks(set())
            if red_light is not None:
                red_light.prune_missing_tracks(set())
            return annotated, violations

        track_ids = boxes.id.int().cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        visible_ids = set(track_ids)
        self.rule.prune_missing_tracks(visible_ids)
        if red_light is not None:
            red_light.prune_missing_tracks(visible_ids)
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
            if self.label_positions is not None:
                if vehicle_type is None:
                    continue
                red = active
                color = (190, 190, 190) if lane_name is None or wrong else (65, 200, 85)
                if red:
                    label += " | SAI LAN"
            else:
                if red_light is not None and vehicle_type is None:
                    continue
                red = self.rule.has_violation(track_id)
                color = (0, 255, 0)
                if red_light is not None and not self.rule.lanes:
                    label = f"{VEHICLE_LABELS[vehicle_type]} | ID {track_id}"
            if red_light is not None and track_id in red_light.violated_ids:
                red = True
                label += " | VUOT DEN DO"
            if red:
                color = (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3 if red else 2)
            tags.append((red, pixel_bbox, label, color))

        for _, bbox, label, text_color in sorted(tags, key=lambda tag: not tag[0]):
            draw_vehicle_tag(annotated, label, bbox, occupied, text_color)
        return annotated, violations

    def _save_evidence(self, original, canvas, bbox, track_id, vehicle_type, t_sec,
                       violation_type=WRONG_LANE):
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
        draw_id(full, f"VIOLATION: {violation_type} (ID:{track_id})", x1, y1)
        prefix = f"{self.session_id}_{track_id}_{violation_type}"
        full_name = f"{prefix}_full.jpg"
        crop_name = f"{prefix}_crop.jpg"
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        for name, image in ((full_name, full), (crop_name, crop)):
            success, jpeg = cv2.imencode(".jpg", image)
            if not success:
                raise OSError(f"Khong ma hoa duoc anh: {name}")
            (EVIDENCE_DIR / name).write_bytes(jpeg.tobytes())

        return {
            "sessionId": self.session_id,
            "trackId": track_id,
            "vehicleType": vehicle_type,
            "violationType": violation_type,
            "timestampInVideo": round(t_sec, 2),
            "fullImageUrl": f"/uploads/evidence/{full_name}",
            "cropImageUrl": f"/uploads/evidence/{crop_name}",
        }


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


def pick_lane(frame):
    height, width = frame.shape[:2]
    scale = min(1.0, 1150 / width, 650 / height)
    preview = cv2.resize(frame, (round(width * scale), round(height * scale)))
    preview_height, preview_width = preview.shape[:2]
    points = []
    title = "Bam 4 goc lan xe | Enter: Xac nhan | R: Chon lai | Esc: Thoat"

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
                print("Hay bam 4 goc theo thu tu quanh vien lan. Nhan R de chon lai.")
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
    if "redLight" in config:
        config["stopLine"] = [[x * scale_x, y * scale_y] for x, y in config["stopLine"]]
        settings = config["redLight"]
        settings["lamps"] = {
            name: [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            for name, (x1, y1, x2, y2) in settings["lamps"].items()
        }
        settings["lineMargin"] = settings.get("lineMargin", 3) * scale_y
    return config


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
        else:
            config = load_roi_config(args.roi, frame.shape)
        red_light_enabled = "redLight" in config
        if not config.get("lanes") and not red_light_enabled:
            raise ValueError("Cau hinh can co lan duong hoac cum den va vach dung.")

        detector = TrafficDetector(f"demo_{uuid4().hex[:12]}", config, device=args.device)
        mode = "red_light" if red_light_enabled else "wrong_lane"
        output = RUNS_DIR / f"ai2_{mode}" / f"{args.video.stem}_{mode}.mp4"
        writer = create_writer(output, fps, frame)
        frame_idx = 0
        total_violations = 0
        while ok and (args.seconds is None or frame_idx / fps < args.seconds):
            annotated, violations = detector.process_frame(frame, frame_idx, fps)
            writer.write(annotated)
            for violation in violations:
                print(f"{violation['violationType']} | ID {violation['trackId']} | "
                      f"{violation['vehicleType']} | {violation['timestampInVideo']:.2f}s", flush=True)
            total_violations += len(violations)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Da xu ly {frame_idx} khung hinh.", flush=True)
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    summary = "Tong so loi" if red_light_enabled else "Tong so loi sai lan"
    print(f"{summary}: {total_violations}")
    print(f"Da luu video: {output}")


def main():
    parser = argparse.ArgumentParser(description="Theo doi xe va thu quy tac giao thong.")
    parser.add_argument("--video", type=Path, default=TEST_VIDEO)
    parser.add_argument("--device", default="cpu")
    lane_source = parser.add_mutually_exclusive_group()
    lane_source.add_argument("--pick-lane", action="store_true", help="Bam 4 goc lan tren anh.")
    lane_source.add_argument("--roi", type=Path, help="Duong dan file JSON theo cau truc roiConfig.")
    parser.add_argument("--allowed", choices=VEHICLE_TYPES, nargs="+",
                        help="Loai xe duoc di trong lan chon bang --pick-lane.")
    parser.add_argument("--seconds", type=float, help="Gioi han so giay video de thu sai lan.")
    args = parser.parse_args()
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds phai lon hon 0.")
    if args.pick_lane and args.allowed is None:
        parser.error("--pick-lane can them --allowed de khai bao loai xe duoc phep.")
    if args.pick_lane or args.roi:
        run_wrong_lane_demo(args)
    else:
        run_tracking_demo(args.video, args.device)


if __name__ == "__main__":
    main()
