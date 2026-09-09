# /// script
# dependencies = [
#   "numpy",
#   "scipy",
#   "norfair",
#   "norfair_rs",
#   "opencv-python",
# ]
# ///

import time

import cv2
import norfair as nf_py
import norfair_rs as nf_rs
import numpy as np
from scipy.spatial.distance import cdist

# ==========================================
#               CONFIGURATION
# ==========================================
WINDOW_NAME = "Norfair Py vs Rs: Decoupled Rendering"
WIDTH, HEIGHT = 800, 600
BG_COLOR = (40, 40, 40)

# Tracker Settings
DIST_THRESH = 50
HIT_MAX = 30
INIT_DELAY = 3
STABILITY_THRESH = 15

# Colors
COL_EXACT_MATCH = (0, 100, 0)  # Dark Green
COL_STABLE_MAP = (0, 255, 0)  # Bright Green
COL_WEAK_MATCH = (0, 255, 255)  # Yellow
COL_CONFLICT = (0, 0, 255)  # Red
COL_PY_ONLY = (200, 200, 200)  # Gray
COL_RS_ONLY = (255, 100, 100)  # Light Red
COL_GT = (60, 60, 60)  # Faint Gray
COL_TEXT = (220, 220, 220)


# ==========================================
#            LOGIC: STABILITY TRACKER
# ==========================================
class StabilityTracker:
    def __init__(self):
        self.mapping_history = {}

    def update(self, matches):
        for py_id, rs_id in matches:
            if py_id not in self.mapping_history:
                self.mapping_history[py_id] = {"rs_id": rs_id, "streak": 1}
            else:
                record = self.mapping_history[py_id]
                if record["rs_id"] == rs_id:
                    record["streak"] += 1
                else:
                    record["rs_id"] = rs_id
                    record["streak"] = 1

    def get_status(self, py_id, rs_id):
        if py_id not in self.mapping_history:
            return COL_WEAK_MATCH, " (New)"

        record = self.mapping_history[py_id]
        if record["rs_id"] != rs_id:
            return COL_CONFLICT, " (Conflict)"

        is_stable = record["streak"] >= STABILITY_THRESH
        is_exact = py_id == rs_id

        if is_exact:
            return (COL_EXACT_MATCH, "") if is_stable else (COL_WEAK_MATCH, f" ({record['streak']})")
        else:
            return (COL_STABLE_MAP, " (Map)") if is_stable else (COL_WEAK_MATCH, f" ({record['streak']})")


# ==========================================
#            SIMULATION LOGIC
# ==========================================
class SimEntity:
    def __init__(self, x, y, is_box):
        self.x, self.y = float(x), float(y)
        self.is_box = is_box
        self.vx, self.vy = np.random.uniform(-1, 1, 2)
        norm = np.linalg.norm([self.vx, self.vy])
        if norm == 0:
            norm = 1
        self.vx /= norm
        self.vy /= norm
        self.w, self.h = (40, 40) if is_box else (0, 0)

    def update(self, bounds, step_size):
        dx = self.vx * step_size * 2.0
        dy = self.vy * step_size * 2.0
        self.x += dx
        self.y += dy

        max_x = bounds[0] - self.w
        max_y = bounds[1] - self.h

        if self.x < 0:
            self.x = 0
            self.vx *= -1
        elif self.x > max_x:
            self.x = max_x
            self.vx *= -1
        if self.y < 0:
            self.y = 0
            self.vy *= -1
        elif self.y > max_y:
            self.y = max_y
            self.vy *= -1

    def get_points(self):
        if self.is_box:
            return np.array([[self.x, self.y], [self.x + self.w, self.y + self.h]])
        else:
            return np.array([[self.x, self.y], [self.x, self.y]])


# ==========================================
#           VISUALIZATION HELPERS
# ==========================================
def draw_text_outline(frame, text, pos, font_scale, color, thickness):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_track_result(frame, points, color, label_txt, thickness=2):
    pts = points.astype(int)
    is_point = np.linalg.norm(pts[0] - pts[1]) < 2

    if is_point:
        center = pts[0]
        cv2.circle(frame, tuple(center), 6, color, thickness)
        pt_lbl = (center[0] + 12, center[1] + 5)
    else:
        tl, br = pts[0], pts[1]
        cv2.rectangle(frame, tuple(tl), tuple(br), color, thickness)
        pt_lbl = (tl[0], tl[1] - 8)

    if label_txt:
        draw_text_outline(frame, label_txt, pt_lbl, 0.5, color, 1)


def compare_tracks_and_draw(frame, tracked_py, tracked_rs, stability_tracker):
    py_objs = [t for t in tracked_py if t.estimate is not None]
    rs_objs = [t for t in tracked_rs if t.estimate is not None]

    matches = []
    matched_rs = set()

    if py_objs and rs_objs:
        py_pts = [t.estimate.mean(axis=0) for t in py_objs]
        rs_pts = [t.estimate.mean(axis=0) for t in rs_objs]
        dists = cdist(py_pts, rs_pts)

        for i, py_obj in enumerate(py_objs):
            best_idx = np.argmin(dists[i])
            if dists[i][best_idx] < 20:
                rs_obj = rs_objs[best_idx]
                matches.append((py_obj, rs_obj))
                matched_rs.add(rs_obj.id)
            else:
                draw_track_result(frame, py_obj.estimate, COL_PY_ONLY, f"Py:{py_obj.id}")
    elif py_objs:
        for py_obj in py_objs:
            draw_track_result(frame, py_obj.estimate, COL_PY_ONLY, f"Py:{py_obj.id}")

    stability_tracker.update([(p.id, r.id) for p, r in matches])

    for py_obj, rs_obj in matches:
        color, suffix = stability_tracker.get_status(py_obj.id, rs_obj.id)
        label = f"P{py_obj.id}:R{rs_obj.id}{suffix}"
        draw_track_result(frame, py_obj.estimate, color, label, thickness=2)

    for rs_obj in rs_objs:
        if rs_obj.id not in matched_rs:
            draw_track_result(frame, rs_obj.estimate, COL_RS_ONLY, f"Rs:{rs_obj.id}")


# ==========================================
#                 MAIN LOOP
# ==========================================
def main():
    def create_trackers():
        cfg = dict(
            distance_function="mean_euclidean",
            distance_threshold=DIST_THRESH,
            hit_counter_max=HIT_MAX,
            initialization_delay=INIT_DELAY,
            pointwise_hit_counter_max=10,
        )
        return nf_py.Tracker(**cfg), nf_rs.Tracker(**cfg), StabilityTracker()

    tracker_py, tracker_rs, stability_tracker = create_trackers()
    entities = []

    cv2.namedWindow(WINDOW_NAME)

    def nothing(x):
        pass

    cv2.createTrackbar("Count", WINDOW_NAME, 10, 50, nothing)
    cv2.createTrackbar("Step Size", WINDOW_NAME, 3, 20, nothing)
    cv2.createTrackbar("Sim FPS", WINDOW_NAME, 60, 120, nothing)  # Only controls SIMULATION updates
    cv2.createTrackbar("Drop %", WINDOW_NAME, 10, 99, nothing)
    cv2.createTrackbar("Sync Drops", WINDOW_NAME, 1, 1, nothing)
    cv2.createTrackbar("Reset", WINDOW_NAME, 0, 1, nothing)

    # Timing Variables
    last_sim_time = time.time()
    last_render_time = time.time()

    # Render FPS Target
    RENDER_FPS = 60.0
    render_interval = 1.0 / RENDER_FPS

    # Placeholders for visualization (to persist between render frames if sim is slow)
    tracked_py, tracked_rs = [], []
    t_py, t_rs = 0.0, 0.0

    print("UI is decoupled. Rendering at 60 FPS.")

    while True:
        current_time = time.time()

        # --- 1. Simulation Loop (Variable FPS) ---
        target_sim_fps = cv2.getTrackbarPos("Sim FPS", WINDOW_NAME)
        if target_sim_fps < 1:
            target_sim_fps = 1
        sim_interval = 1.0 / target_sim_fps

        if current_time - last_sim_time >= sim_interval:
            last_sim_time = current_time

            # Get GUI Inputs
            target_n = cv2.getTrackbarPos("Count", WINDOW_NAME)
            step_size = cv2.getTrackbarPos("Step Size", WINDOW_NAME)
            drop_prob = cv2.getTrackbarPos("Drop %", WINDOW_NAME) / 100.0
            sync_drops = cv2.getTrackbarPos("Sync Drops", WINDOW_NAME) == 1

            # Reset Logic
            if cv2.getTrackbarPos("Reset", WINDOW_NAME) == 1:
                tracker_py, tracker_rs, stability_tracker = create_trackers()
                entities = []
                cv2.setTrackbarPos("Reset", WINDOW_NAME, 0)

            # Adjust Entity Count
            if len(entities) < target_n:
                is_box = np.random.rand() > 0.5
                entities.append(SimEntity(np.random.randint(0, WIDTH - 50), np.random.randint(0, HEIGHT - 50), is_box))
            elif len(entities) > target_n:
                entities.pop()

            # Update Positions
            raw_points = []
            for ent in entities:
                ent.update((WIDTH, HEIGHT), step_size)
                raw_points.append(ent.get_points())

            # Drops Logic
            n = len(raw_points)
            if n > 0:
                if sync_drops:
                    mask_py = np.random.rand(n) >= drop_prob
                    mask_rs = mask_py
                else:
                    mask_py = np.random.rand(n) >= drop_prob
                    mask_rs = np.random.rand(n) >= drop_prob
            else:
                mask_py, mask_rs = [], []

            det_py = [nf_py.Detection(pts) for i, pts in enumerate(raw_points) if mask_py[i]]
            det_rs = [nf_rs.Detection(pts) for i, pts in enumerate(raw_points) if mask_rs[i]]

            # Run Trackers
            t0 = time.perf_counter()
            tracked_py = tracker_py.update(detections=det_py)
            t_py = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            tracked_rs = tracker_rs.update(detections=det_rs)
            t_rs = (time.perf_counter() - t0) * 1000

        # --- 2. Render Loop (Fixed 60 FPS) ---
        if current_time - last_render_time >= render_interval:
            last_render_time = current_time

            frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)

            # Draw Ground Truth
            for ent in entities:
                draw_track_result(frame, ent.get_points(), COL_GT, "", thickness=1)

            # Draw Last Known Tracker State
            compare_tracks_and_draw(frame, tracked_py, tracked_rs, stability_tracker)

            # UI Stats
            draw_text_outline(frame, f"Py: {t_py:.2f}ms", (10, 30), 0.6, COL_TEXT, 1)
            draw_text_outline(frame, f"Rs: {t_rs:.2f}ms", (10, 55), 0.6, COL_TEXT, 1)

            status = "SYNCED" if cv2.getTrackbarPos("Sync Drops", WINDOW_NAME) else "UNSYNCED"
            drop_p = int(cv2.getTrackbarPos("Drop %", WINDOW_NAME))
            sim_fps_val = cv2.getTrackbarPos("Sim FPS", WINDOW_NAME)
            draw_text_outline(frame, f"Drop: {drop_p}% [{status}]", (10, 80), 0.6, COL_TEXT, 1)
            draw_text_outline(frame, f"Sim FPS: {sim_fps_val}", (10, 105), 0.6, COL_TEXT, 1)

            # Legend
            lx = WIDTH - 200
            draw_text_outline(frame, "Exact Match (ID==ID)", (lx, 30), 0.5, COL_EXACT_MATCH, 1)
            draw_text_outline(frame, "Stable Map (ID!=ID)", (lx, 50), 0.5, COL_STABLE_MAP, 1)
            draw_text_outline(frame, "Weak/New Match", (lx, 70), 0.5, COL_WEAK_MATCH, 1)
            draw_text_outline(frame, "Conflict", (lx, 90), 0.5, COL_CONFLICT, 1)

            cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
