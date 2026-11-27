from collections import deque
import math
import unicodedata

import cv2
import numpy as np

VEHICLE_TYPES = ("motorbike", "car", "truck", "bus")
WRONG_LANE = "WRONG_LANE"
RED_LIGHT = "RED_LIGHT"
SPEEDING = "SPEEDING"
VEHICLE_ALIASES = {
    "car": "car", "cars": "car", "oto": "car", "xeoto": "car", "automobile": "car",
    "truck": "truck", "trucks": "truck", "xetai": "truck", "lorry": "truck",
    "motorbike": "motorbike", "motorbikes": "motorbike", "motorcycle": "motorbike",
    "motorcycles": "motorbike", "xemay": "motorbike", "scooter": "motorbike",
    "bus": "bus", "buses": "bus", "xebus": "bus", "xebuyt": "bus",
}


def normalize_vehicle_type(name):
    decomposed = unicodedata.normalize("NFKD", str(name).casefold().replace("đ", "d"))
    key = "".join(char for char in decomposed if char.isascii() and char.isalnum())
    return VEHICLE_ALIASES.get(key)


class WrongLaneRule:
    def __init__(self, roi_config):
        self.sample_style = roi_config.get("sampleOverlay")
        self.lanes = []
        for index, lane in enumerate(roi_config.get("lanes") or [], start=1):
            name = str(lane.get("laneName") or f"Lan {index}")
            points = np.asarray(lane["polygon"], dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
                raise ValueError(f"{name}: polygon can it nhat 3 diem [x, y].")
            if not np.isfinite(points).all() or cv2.contourArea(points) <= 0:
                raise ValueError(f"{name}: polygon khong hop le.")
            allowed = {normalize_vehicle_type(value) for value in lane["allowedVehicles"]}
            if not allowed or None in allowed:
                raise ValueError(f"{name}: allowedVehicles phai chua loai xe hop le.")
            self.lanes.append((name, points, allowed))

        self.track_history = {}
        self.last_seen = {}
        self.wrong_streak = {}
        self.violated_records = set()

    def observe(self, track_id, bbox, vehicle_type, frame_idx):
        _, _, active = self.observe_frame(track_id, bbox, vehicle_type, frame_idx, 30.0)
        return active and not self.has_violation(track_id)

    def observe_frame(self, track_id, bbox, vehicle_type, frame_idx, fps, frame_shape=None):
        x1, _, x2, y2 = bbox
        anchor = ((x1 + x2) / 2, y2)
        if self.last_seen.get(track_id) != frame_idx - 1:
            self.track_history[track_id] = deque(maxlen=25)
            self.wrong_streak.pop(track_id, None)
        history = self.track_history[track_id]
        history.append(anchor)
        self.last_seen[track_id] = frame_idx

        lane_name, allowed_types = self.lane_at(anchor, frame_shape)
        wrong = lane_name is not None and vehicle_type is not None and vehicle_type not in allowed_types
        if self.sample_style is None:
            return lane_name, wrong, wrong and len(history) >= 5
        if not wrong:
            self.wrong_streak.pop(track_id, None)
            return lane_name, False, False

        streak = self.wrong_streak.get(track_id, 0) + 1
        self.wrong_streak[track_id] = streak
        required = max(1, math.ceil(self.sample_style["confirmSeconds"] * fps))
        return lane_name, True, streak >= required

    def lane_at(self, anchor, frame_shape):
        for name, polygon, allowed_types in self.lanes:
            if self.sample_style is None:
                inside = cv2.pointPolygonTest(polygon, anchor, False) >= 0
            else:
                x, y = anchor
                if frame_shape is not None and (
                    not 0 <= x < frame_shape[1] or y >= frame_shape[0] - 2
                ):
                    break
                margin = self.sample_style["sideMargin"]
                inside = (
                    cv2.pointPolygonTest(polygon, (x - margin, y), False) > 0
                    and cv2.pointPolygonTest(polygon, (x + margin, y), False) > 0
                )
            if inside:
                return name, allowed_types
        return None, ()

    def has_violation(self, track_id):
        return (track_id, WRONG_LANE) in self.violated_records

    def mark_violation(self, track_id):
        self.violated_records.add((track_id, WRONG_LANE))

    def prune_missing_tracks(self, visible_ids):
        for track_id in self.last_seen.keys() - visible_ids:
            del self.last_seen[track_id]
            del self.track_history[track_id]
            self.wrong_streak.pop(track_id, None)


class RedLightRule:
    def __init__(self, stop_line, settings):
        self.points = np.asarray(stop_line, dtype=float)
        if (self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 2
                or not np.isfinite(self.points).all()):
            raise ValueError("stopLine can it nhat hai diem [x, y] hop le.")
        vectors = np.diff(self.points, axis=0)
        self.segment_lengths = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(self.segment_lengths).all() or np.any(self.segment_lengths <= 0):
            raise ValueError("Cac diem lien tiep cua stopLine phai phan biet.")
        self.tangents = vectors / self.segment_lengths[:, None]
        self.normals = np.column_stack((-self.tangents[:, 1], self.tangents[:, 0]))
        main_segment = int(np.argmax(self.segment_lengths))
        self.main_tangent = self.tangents[main_segment]
        self.main_normal = self.normals[main_segment]
        self.offsets = (self.points - self.points[0]) @ self.main_tangent
        if np.any(np.diff(self.offsets) <= 0):
            raise ValueError("Cac doan stopLine phai noi tiep theo huong cua doan chinh.")
        self.lamps = {color: tuple(round(value) for value in settings["lamps"][color])
                      for color in ("red", "yellow", "green")}
        self.cycle = settings.get("simulatedCycle")
        if self.cycle is not None:
            self.cycle = tuple(float(self.cycle[color]) for color in ("red", "green", "yellow"))
            if not all(math.isfinite(value) and value > 0 for value in self.cycle):
                raise ValueError("Thoi gian den mo phong phai la so duong.")
        self.margin = float(settings.get("lineMargin", 3))
        self.stable_frames = max(1, int(settings.get("stableFrames", 3)))
        self.state = "unknown"
        self.raw_state = "unknown"
        self.signal_frame = None
        self.streak = 0
        self.red_start = None
        self.previous_side = {}
        self.pending_crossings = {}
        self.last_seen = {}
        self.violated_ids = set()

    def read_color(self, frame):
        height, width = frame.shape[:2]
        detected = []
        for color, (x1, y1, x2, y2) in self.lamps.items():
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                return "unknown"
            hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv)
            bright = (saturation >= 105) & (value >= 170)
            if color == "red":
                # The upper lamp can look orange when it is overexposed.
                matches = (hue <= 23) | (hue >= 165)
            elif color == "yellow":
                matches = (hue >= 10) & (hue <= 42)
            else:
                matches = (hue >= 38) & (hue <= 100)
            if np.mean(bright & matches) >= 0.14:
                detected.append(color)
        return detected[0] if len(detected) == 1 else "unknown"

    def update_signal(self, frame, frame_idx, fps=30.0):
        if self.cycle is not None:
            red, green, yellow = self.cycle
            cycle_length = red + green + yellow
            elapsed = frame_idx / fps
            phase = elapsed % cycle_length
            previous_cycle_start = self.red_start
            if phase < red:
                self.state = "red"
                self.red_start = math.floor(elapsed / cycle_length) * cycle_length * fps
            else:
                self.state = "green" if phase < red + green else "yellow"
                self.red_start = None
            if self.red_start is None or self.red_start != previous_cycle_start:
                self.pending_crossings.clear()
            return

        color = self.read_color(frame)
        same_phase = color == self.raw_state and self.signal_frame == frame_idx - 1
        if not same_phase:
            self.pending_crossings.clear()
        self.streak = self.streak + 1 if same_phase else 1
        self.raw_state = color
        self.signal_frame = frame_idx
        self.red_start = frame_idx - self.streak + 1 if color == "red" else None
        self.state = color if self.streak >= self.stable_frames else "unknown"

    def observe(self, track_id, bbox, frame_idx, fps):
        if track_id in self.violated_ids:
            return None
        previous = self.last_seen.get(track_id)
        if previous is None or previous[2] != frame_idx - 1:
            previous = None
            self.previous_side.pop(track_id, None)
            self.pending_crossings.pop(track_id, None)

        x1, _, x2, y2 = bbox
        point = np.asarray(((x1 + x2) / 2, y2))
        distance = self._signed_distance(point)
        self.last_seen[track_id] = (point, distance, frame_idx)
        side = self.previous_side.get(track_id)

        if distance >= 0:
            self.pending_crossings.pop(track_id, None)
        elif (self.red_start is not None and previous is not None and previous[1] >= 0
              and side is not None and side[1] > 0):
            crossing = self._crossing_frame(previous, point, frame_idx)
            if crossing is not None and crossing >= self.red_start:
                self.pending_crossings[track_id] = crossing

        if abs(distance) > self.margin:
            if distance < 0 and side is not None and side[1] > 0:
                motion = point - side[0]
                forward = -float(np.dot(motion, self.main_normal))
                sideways = abs(float(np.dot(motion, self.main_tangent)))
                if forward <= 0 or sideways > 1.4 * forward:
                    self.pending_crossings.pop(track_id, None)
            side = self.previous_side[track_id] = (point, distance)

        # Keep crossings from the first red frames until the lamp is confirmed.
        pending = self.pending_crossings.get(track_id)
        if self.state == "red" and pending is not None and side is not None and side[1] < 0:
            return pending / fps
        return None

    def _signed_distance(self, point):
        along = float(np.dot(point - self.points[0], self.main_tangent))
        index = int(np.searchsorted(self.offsets[1:-1], along, side="right"))
        return float(np.dot(point - self.points[index], self.normals[index]))

    def _crossing_frame(self, previous, point, frame_idx):
        old_point, _, old_frame = previous
        motion = point - old_point
        intersections = []
        epsilon = 1e-9
        for start, tangent, normal, length in zip(
                self.points[:-1], self.tangents, self.normals, self.segment_lengths):
            forward = -float(np.dot(motion, normal))
            if forward <= epsilon:
                continue
            offset = old_point - start
            fraction = float(np.dot(offset, normal)) / forward
            if -epsilon <= fraction <= 1 + epsilon:
                along = float(np.dot(offset + fraction * motion, tangent))
                if -epsilon <= along <= length + epsilon:
                    intersections.append(max(0.0, min(1.0, fraction)))
        if not intersections:
            return None
        fraction = min(intersections)
        return old_frame + (frame_idx - old_frame) * fraction

    def mark_violation(self, track_id):
        self.violated_ids.add(track_id)
        self.pending_crossings.pop(track_id, None)
        self.last_seen.pop(track_id, None)
        self.previous_side.pop(track_id, None)

    def prune_missing_tracks(self, visible_ids):
        for track_id in self.last_seen.keys() - visible_ids:
            self.last_seen.pop(track_id)
            self.previous_side.pop(track_id, None)
            self.pending_crossings.pop(track_id, None)


class SpeedingRule:
    def __init__(self, limit_kmh):
        if not math.isfinite(limit_kmh) or limit_kmh <= 0:
            raise ValueError("Gioi han toc do phai lon hon 0.")
        self.limit_kmh = limit_kmh
        self.above_limit = {}
        self.violated_ids = set()

    def observe(self, track_id, speed_kmh, frame_idx, fps):
        if speed_kmh is None or not math.isfinite(speed_kmh) or speed_kmh <= self.limit_kmh:
            self.above_limit.pop(track_id, None)
            return False

        first_frame, last_frame = self.above_limit.get(track_id, (frame_idx, frame_idx - 1))
        if last_frame != frame_idx - 1:
            first_frame = frame_idx
        self.above_limit[track_id] = (first_frame, frame_idx)
        return frame_idx - first_frame >= math.ceil(0.5 * fps)

    def mark_violation(self, track_id):
        self.violated_ids.add(track_id)

    def prune_missing_tracks(self, visible_ids):
        for track_id in self.above_limit.keys() - visible_ids:
            del self.above_limit[track_id]
