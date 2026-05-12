import time
import os
import math
import random
import traceback
import cv2
from flask import app
import requests
#import torch
import gdown
import cv2
from collections import deque

from PIL import Image
from datetime import datetime, timezone

import joblib
import pandas as pd
import numpy as np
import re

import firebase_admin
from firebase_admin import credentials, firestore


# =========================================================
# CONFIG
# =========================================================
BASE_PATH = os.getcwd()

TEMP_DIR = os.path.join(BASE_PATH, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

MODELS_DIR = os.path.join(BASE_PATH, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


FIREBASE_CRED_PATH = os.path.join(
    BASE_PATH,
    "firebase",
    "roomvisualizer-f206e-firebase-adminsdk-fbsvc-ca71a15e25.json"
)

# =========================================================
# TRUE MULTIROOM v6
# =========================================================
DEPTH_MODEL_PATH = os.path.join(
    BASE_PATH,
    "models",
    "layout_depth_model_v6_MULTIROOM.pkl"
)

SCALE_MODEL_PATH = os.path.join(
    BASE_PATH,
    "models",
    "layout_scale_model_v6_MULTIROOM.pkl"
)
DEPTH_MODEL_FILE_ID = "1AIJkQYr6vZU7j9JgKY1A6bLQZi7z_Urx"
SCALE_MODEL_FILE_ID = "146ZSMoI6pazZbTl36sNK3U_G7Y6wioY6"

DATA_DIR = os.path.join(BASE_PATH, "data")

EXPANDED_DATA_PATH = os.path.join(
    DATA_DIR,
    "master_training_data_layout_v6_expanded_MULTIROOM.csv"
)

DATA_PATH = os.path.join(
    DATA_DIR,
    "master_training_data_layout_v6_TRUE_MULTIROOM.csv"
)

KB_METADATA_PATH = os.path.join(
    DATA_DIR,
    "kitchen_bathroom_metadata_v4_clean.csv"
)

POLL_SECONDS = 20

def download_file_if_missing(file_id, output_path):
    if os.path.exists(output_path):
        print(f"✅ Exists: {output_path}")
        return

    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"⬇️ Downloading {output_path}...")

    gdown.download(
        url,
        output_path,
        quiet=False
    )

    print(f"✅ Downloaded: {output_path}")
# =========================================================
# FIREBASE SETUP
# =========================================================
cred = credentials.Certificate(FIREBASE_CRED_PATH)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

request_collection = db.collection("room_designs").document("input_data").collection("requests")
prediction_collection = db.collection("room_designs").document("input_data").collection("predictions")

smart_metadata_collection = (
    db.collection("smart_model_metadata")
)

SMART_METADATA_CACHE = {}
# =========================================================
# LOAD MODELS + DATA
# =========================================================
depth_model = None
scale_model = None

def get_models():
    global depth_model, scale_model

    if not os.path.exists(DEPTH_MODEL_PATH):
        print("⬇️ Downloading depth model...")
        download_file_if_missing(
            DEPTH_MODEL_FILE_ID,
            DEPTH_MODEL_PATH
        )

    if not os.path.exists(SCALE_MODEL_PATH):
        print("⬇️ Downloading scale model...")
        download_file_if_missing(
            SCALE_MODEL_FILE_ID,
            SCALE_MODEL_PATH
        )

    if depth_model is None:
        print("Loading depth model...")
        depth_model = joblib.load(DEPTH_MODEL_PATH)

    if scale_model is None:
        print("Loading scale model...")
        scale_model = joblib.load(SCALE_MODEL_PATH)

    return depth_model, scale_model

# =========================================================
# MIDAS DEPTH ESTIMATION
# =========================================================
MIDAS_DEVICE = "cpu"


midas = None
midas_transform = None


def load_midas():
    import torch
    global midas
    global midas_transform

    if midas is not None:
        return

    print("Loading MiDaS...")

    midas_model = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small"
    )

    device = torch.device("cpu")
    midas_model.to(device)

    midas_model.eval()

    transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms"
    )

    midas_small_transform = transforms.small_transform

    midas = midas_model
    midas_transform = midas_small_transform

    print("MiDaS ready")

df_master = None
kb_df = None

def load_datasets():
    global df_master, kb_df

    if df_master is not None:
        return

    print("📦 Loading datasets...")

    if os.path.exists(EXPANDED_DATA_PATH):
        df_master = pd.read_csv(
            EXPANDED_DATA_PATH,
            low_memory=True,
            engine="python"
        )
        print("✅ Loaded expanded dataset")
    else:
        df_master = pd.read_csv(DATA_PATH, low_memory=False)
        print("✅ Loaded fallback dataset")

    if os.path.exists(KB_METADATA_PATH):
        kb_df = pd.read_csv(
            KB_METADATA_PATH,
            low_memory=False
        )
    else:
        kb_df = df_master.copy()

    print("✅ Datasets ready")


def normalize_room_label_app(value):
    if value is None or pd.isna(value):
        return value

    value = str(value).strip()

    fixes = {
        "Kitchen": "Kitchen Room",
        "KitchenRoom": "Kitchen Room",
        "Kitchen room": "Kitchen Room",
        "Kitchen Room": "Kitchen Room",
        "Kitchen Room Room": "Kitchen Room",
        "Bathroom Room": "Bathroom",
        "Bath Room": "Bathroom",
        "Bath": "Bathroom",
        "Toilet": "Bathroom",
        "LivingRoom": "Living Room",
        "DiningRoom": "Dining Room",
        "BedRoom": "Bedroom",
    }

    return fixes.get(value, value)

load_datasets()
if "room_type" in df_master.columns:
    df_master["room_type"] = df_master["room_type"].apply(normalize_room_label_app)

if "room_types" in df_master.columns:
    def normalize_room_types_app(value):
        if value is None or pd.isna(value):
            return value

        parts = [
            normalize_room_label_app(v.strip())
            for v in str(value).split("|")
            if v.strip()
        ]

        seen = set()
        cleaned = []

        for p in parts:
            if p not in seen:
                cleaned.append(p)
                seen.add(p)

        return "|".join(cleaned)

    df_master["room_types"] = df_master["room_types"].apply(normalize_room_types_app)

# Load Kitchen/Bathroom metadata.
if os.path.exists(KB_METADATA_PATH):
    kb_df = pd.read_csv(
        KB_METADATA_PATH,
        low_memory=False
    )
    print("✅ Loaded legacy kitchen/bathroom metadata")
else:
    kb_df = df_master[
        (df_master.get("source_dataset", "") == "merged_model_info_kitchen_bathroom")
        | (df_master.get("super-category", "").isin(["Appliance", "Plumbing"]))
        | (df_master.get("room_types", "").astype(str).str.contains("Kitchen Room|Bathroom", na=False))
    ].copy()
    print("✅ Using Kitchen/Bathroom rows from latest training dataset")

if "usable_room_types" not in kb_df.columns and "room_types" in kb_df.columns:
    kb_df["usable_room_types"] = kb_df["room_types"]

if "category_clean" not in kb_df.columns and "category" in kb_df.columns:
    kb_df["category_clean"] = kb_df["category"].astype(str).str.lower()

if "layout_scale" not in kb_df.columns:
    kb_df["layout_scale"] = 1.0

if "anchor_type" not in kb_df.columns:
    kb_df["anchor_type"] = "secondary"

if "placement_zone" not in kb_df.columns:
    kb_df["placement_zone"] = "general"

print(f"✅ Kitchen/Bathroom metadata rows: {len(kb_df)}")
print(f"✅ Phase 3 aligned app loaded")
print(f"✅ Dataset rows: {len(df_master)}")


# =========================================================
# HELPERS
# =========================================================
def clamp01(v):
    return max(0.0, min(1.0, float(v)))


def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_str(value, default="unknown"):
    if value is None or pd.isna(value):
        return default
    return str(value)


def load_smart_metadata(model_id):
    global SMART_METADATA_CACHE

    model_id = str(model_id).strip()

    if model_id in SMART_METADATA_CACHE:
        return SMART_METADATA_CACHE[model_id]

    try:
        doc = (
            smart_metadata_collection
            .document(model_id)
            .get()
        )

        if not doc.exists:
            SMART_METADATA_CACHE[model_id] = None
            return None

        data = doc.to_dict()
        SMART_METADATA_CACHE[model_id] = data
        print(f"[SMART] loaded metadata for {model_id}")
        return data

    except Exception as e:
        print(f"[SMART] metadata load failed: {e}")
        return None


def snap_rotation_90(angle):
    try:
        angle = float(angle) % 360.0
    except Exception:
        angle = 0.0
    return min([0.0, 90.0, 180.0, 270.0], key=lambda x: abs(x - angle))


def enforce_code_minimums(width, length, height, room_type):
    width = float(width)
    length = float(length)
    height = float(height)

    if room_type in ["Bedroom", "Living Room", "Dining Room"]:
        min_dim = 2.0
        min_area = 6.0
    elif room_type in ["Kitchen Room", "Kitchen"]:
        min_dim = 1.5
        min_area = 3.0
    elif room_type in ["Bathroom", "Toilet", "Bath"]:
        min_dim = 0.9
        min_area = 1.2
    else:
        min_dim = 2.0
        min_area = 6.0

    width = max(width, min_dim)
    length = max(length, min_dim)

    if width * length < min_area:
        scale = (min_area / (width * length)) ** 0.5
        width *= scale
        length *= scale

    height = max(height, 2.4)
    return width, length, height


# FIX: convert_to_screen_space is kept for reference but its output
# (screen_x, screen_y, depth) is NO LONGER written to Firestore.
# Unity uses x_norm/z_norm directly for viewport mapping.
# Removing these fields avoids confusion about which coordinate set to use.
def convert_to_screen_space(x_norm, z_norm):
    near_floor_y = 0.18
    horizon_y = 0.52

    x_norm = clamp01(x_norm)
    z_norm = clamp01(z_norm)

    screen_y = near_floor_y + (horizon_y - near_floor_y) * (z_norm ** 1.6)

    half_w_near = 0.48
    half_w_far = 0.35
    half_w = half_w_near + (half_w_far - half_w_near) * z_norm

    x_left = 0.50 - half_w
    x_right = 0.50 + half_w
    screen_x = x_left + x_norm * (x_right - x_left)

    depth = 0.05 + (z_norm ** 1.8) * 0.95

    screen_x = max(x_left, min(x_right, screen_x))
    screen_y = max(near_floor_y, min(horizon_y, screen_y))
    depth = max(0.05, min(0.95, depth))

    return round(screen_x, 3), round(screen_y, 3), round(depth, 3)


def compute_y_norm(role, category, room_height):
    category_l = str(category).lower()

    if "pendant lamp" in category_l:
        return clamp(
            0.94 + (room_height - 2.7) * 0.05,
            0.88,
            0.99
        )

    if "ceiling lamp" in category_l:
        return clamp(
            0.90 + (room_height - 2.7) * 0.04,
            0.84,
            0.98
        )

    if "wall lamp" in category_l:
        return clamp(
            0.58 + (room_height - 2.7) * 0.05,
            0.45,
            0.80
        )

    if "floor lamp" in category_l:
        return 0.0

    if role == "light":
        return clamp(0.72, 0.50, 0.90)

    return 0.0


# =========================================================
# IMAGE UNDERSTANDING + DEPTH ESTIMATION
# =========================================================
def get_default_scene_meta():
    return {
        "horizon_y": 0.42,
        "back_wall_center_x": 0.50,
        "usable_left_x": 0.12,
        "usable_right_x": 0.88,
        "floor_top_y": 0.40,
        "anchor_wall": "back_wall",
        "camera_bias": "center",
        "room_depth_confidence": 0.0,
        "estimated_room_depth": 5.0,
        "estimated_room_width": 4.0,
        "ceiling_y": 0.08,
    }


# =========================================================
# DOWNLOAD IMAGE
# =========================================================
def download_image_temp(image_url):
    try:
        response = requests.get(image_url, timeout=20)

        if response.status_code != 200:
            return None

        temp_filename = f"room_{int(time.time())}_{random.randint(1000,9999)}.jpg"

        temp_path = os.path.join(
        TEMP_DIR,
        temp_filename
    )

        with open(temp_path, "wb") as f:
            f.write(response.content)

        return temp_path

    except Exception as e:
        print(f"[IMAGE] download failed: {e}")
        return None


# =========================================================
# MIDAS DEPTH MAP
# =========================================================
def estimate_depth_map(image_bgr):
    load_midas()
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        input_batch = midas_transform(image_rgb).to(MIDAS_DEVICE)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        return depth_map

    except Exception as e:
        print(f"[DEPTH] estimation failed: {e}")
        return None


# =========================================================
# FLOOR DETECTION
# =========================================================
def detect_floor_region(depth_map):
    h, w = depth_map.shape
    lower_region = depth_map[int(h * 0.55):, :]
    floor_depth = np.mean(lower_region)

    floor_top_y = 0.45
    if floor_depth > 0.6:
        floor_top_y = 0.52
    elif floor_depth > 0.4:
        floor_top_y = 0.48

    return {
        "floor_top_y": floor_top_y,
        "floor_depth": float(floor_depth),
    }


# =========================================================
# FLOOR MASK SEGMENTATION
# =========================================================
def segment_floor_mask(image_bgr, depth_map):
    h, w = depth_map.shape
    lower_y = int(h * 0.45)
    floor_region = depth_map[lower_y:, :]
    floor_depth_threshold = np.mean(floor_region) * 0.92

    floor_mask = (depth_map >= floor_depth_threshold).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)

    return floor_mask


# =========================================================
# FIND USABLE FLOOR REGIONS
# =========================================================
def extract_usable_regions(floor_mask):
    h, w = floor_mask.shape
    visited = np.zeros_like(floor_mask, dtype=np.uint8)
    usable_regions = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        for x in range(w):
            if floor_mask[y, x] == 0:
                continue
            if visited[y, x]:
                continue

            queue = deque()
            queue.append((x, y))
            visited[y, x] = 1
            region = []

            while queue:
                cx, cy = queue.popleft()
                region.append((cx, cy))

                for dx, dy in directions:
                    nx = cx + dx
                    ny = cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if floor_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = 1
                            queue.append((nx, ny))

            if len(region) > 500:
                usable_regions.append(region)

    return usable_regions


# =========================================================
# COMPUTE USABLE SPACE CENTER
# =========================================================
def compute_room_open_space(usable_regions, image_shape):
    h, w = image_shape[:2]

    if not usable_regions:
        return {
            "usable_center_x": 0.50,
            "usable_floor_y": 0.75,
        }

    largest = max(usable_regions, key=len)
    xs = [p[0] for p in largest]
    ys = [p[1] for p in largest]

    center_x = np.mean(xs) / w
    floor_y = np.mean(ys) / h

    return {
        "usable_center_x": float(center_x),
        "usable_floor_y": float(floor_y),
    }


# =========================================================
# WALL LINE DETECTION
# =========================================================
def detect_room_walls(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=80, minLineLength=120, maxLineGap=20
    )

    wall_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > 100 or dy > 100:
                wall_lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return wall_lines


# =========================================================
# WINDOW DETECTION
# =========================================================
def detect_probable_windows(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 180])
    upper = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    windows = []
    h, w = image_bgr.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4000:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        if 0.5 <= aspect <= 3.5:
            windows.append({
                "x_norm": (x + cw / 2) / w,
                "y_norm": (y + ch / 2) / h,
                "w_norm": cw / w,
                "h_norm": ch / h,
            })

    return windows


# =========================================================
# DOOR DETECTION
# =========================================================
def detect_probable_doors(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lower_region = gray[int(h * 0.35):, :]
    _, thresh = cv2.threshold(lower_region, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    doors = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = ch / max(cw, 1)
        if aspect > 1.5:
            doors.append({
                "x_norm": (x + cw / 2) / w,
                "y_norm": (y + ch / 2) / h,
                "w_norm": cw / w,
                "h_norm": ch / h,
            })

    return doors


# =========================================================
# ROOM GEOMETRY ESTIMATION
# =========================================================
def estimate_room_geometry(depth_map):
    h, w = depth_map.shape
    left_depth = np.mean(depth_map[:, :int(w * 0.2)])
    center_depth = np.mean(depth_map[:, int(w * 0.4):int(w * 0.6)])
    right_depth = np.mean(depth_map[:, int(w * 0.8):])

    room_width_est = 4.0
    room_depth_est = 5.0

    if center_depth > 0.6:
        room_depth_est = 7.0
    elif center_depth > 0.45:
        room_depth_est = 5.5

    if abs(left_depth - right_depth) < 0.05:
        camera_bias = "center"
        center_x = 0.50
    elif left_depth > right_depth:
        camera_bias = "left"
        center_x = 0.58
    else:
        camera_bias = "right"
        center_x = 0.42

    return {
        "estimated_room_width": room_width_est,
        "estimated_room_depth": room_depth_est,
        "camera_bias": camera_bias,
        "back_wall_center_x": center_x,
    }


# =========================================================
# MAIN IMAGE UNDERSTANDING
# =========================================================
def analyze_room_image_metadata(image_url: str):
    meta = get_default_scene_meta()

    if not image_url:
        print("[IMAGE] no image provided")
        return meta

    try:
        local_path = download_image_temp(image_url)

        if not local_path:
            return meta

        image_bgr = cv2.imread(local_path)

        if image_bgr is None:
            return meta

        h, w = image_bgr.shape[:2]
        print(f"[IMAGE] loaded image {w}x{h}")

        depth_map = estimate_depth_map(image_bgr)
        if depth_map is None:
            return meta

        floor_meta = detect_floor_region(depth_map)
        floor_mask = segment_floor_mask(image_bgr, depth_map)
        usable_regions = extract_usable_regions(floor_mask)
        usable_meta = compute_room_open_space(usable_regions, image_bgr.shape)
        geometry_meta = estimate_room_geometry(depth_map)
        wall_lines = detect_room_walls(image_bgr)
        windows = detect_probable_windows(image_bgr)
        doors = detect_probable_doors(image_bgr)

        meta.update(floor_meta)
        meta.update(geometry_meta)
        meta.update(usable_meta)

        meta["room_depth_confidence"] = 0.85
        meta["wall_lines"] = wall_lines
        meta["windows"] = windows
        meta["doors"] = doors

        print("[IMAGE] scene analysis complete")
        print(meta)

        return meta

    except Exception as e:
        print(f"[IMAGE] scene analysis failed: {e}")
        traceback.print_exc()
        return meta


# =========================================================
# ADVANCED ROTATION HELPERS
# =========================================================
def snap_to_cardinal(angle_deg: float) -> float:
    angle_deg = angle_deg % 360.0
    return min([0.0, 90.0, 180.0, 270.0], key=lambda a: abs(a - angle_deg))


def infer_wall_from_position(x_norm: float, z_norm: float, edge_threshold: float = 0.18) -> str:
    left_dist = x_norm
    right_dist = 1.0 - x_norm
    back_dist = z_norm
    front_dist = 1.0 - z_norm

    distances = {
        "left_wall": left_dist,
        "right_wall": right_dist,
        "back_wall": back_dist,
        "front_wall": front_dist,
    }

    nearest_wall = min(distances, key=distances.get)
    if distances[nearest_wall] <= edge_threshold:
        return nearest_wall
    return "general"


def yaw_to_face_target(from_x: float, from_z: float, to_x: float, to_z: float) -> float:
    dx = to_x - from_x
    dz = to_z - from_z

    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0

    angle = math.degrees(math.atan2(dx, dz))
    return snap_to_cardinal(angle)


def get_anchor_target_for_item(category: str, super_category: str, anchors: dict):
    category_l = (category or "").lower()
    super_l = (super_category or "").lower()

    if "chair" in category_l or super_l == "chair":
        return anchors.get("desk") or anchors.get("table") or anchors.get("bed") or anchors.get("sofa")

    if "nightstand" in category_l:
        return anchors.get("bed")

    if "coffee table" in category_l:
        return anchors.get("sofa")

    if "tv stand" in category_l:
        return anchors.get("sofa")

    return None


def get_advanced_rotation(
    category: str,
    super_category: str,
    role: str,
    zone: str,
    x_norm: float,
    z_norm: float,
    anchors: dict
) -> float:
    category_l = (category or "").lower()
    super_l = (super_category or "").lower()
    role_l = (role or "").lower()
    zone_l = (zone or "").lower()

    nearest_wall = infer_wall_from_position(x_norm, z_norm)

    if role_l == "light" or super_l == "lighting":
        if "wall lamp" in category_l:
            if nearest_wall == "left_wall":
                return 90.0
            if nearest_wall == "right_wall":
                return 270.0
            if nearest_wall == "back_wall":
                return 180.0
            return 0.0
        return 0.0

    if "bed" in category_l or super_l == "bed":
        if zone_l == "back_wall" or nearest_wall == "back_wall":
            return 180.0
        if nearest_wall == "front_wall":
            return 0.0
        if nearest_wall == "left_wall":
            return 90.0
        if nearest_wall == "right_wall":
            return 270.0
        return 180.0

    if (
        "wardrobe" in category_l
        or "cabinet" in category_l
        or "shelf" in category_l
        or "bookcase" in category_l
        or "shoe cabinet" in category_l
        or "wine cabinet" in category_l
        or "sideboard" in category_l
        or super_l == "cabinet/shelf/desk"
    ):
        if zone_l == "side_wall" or nearest_wall in ["left_wall", "right_wall"]:
            return 90.0 if nearest_wall == "left_wall" else 270.0
        if zone_l == "back_wall" or nearest_wall == "back_wall":
            return 180.0
        if nearest_wall == "front_wall":
            return 0.0
        return 180.0

    if "desk" in category_l or "dressing table" in category_l:
        if nearest_wall == "left_wall":
            return 90.0
        if nearest_wall == "right_wall":
            return 270.0
        return 90.0

    if "tv stand" in category_l:
        target = anchors.get("sofa")
        if target:
            return yaw_to_face_target(x_norm, z_norm, target["x_norm"], target["z_norm"])
        return 0.0

    if "chair" in category_l or super_l == "chair":
        target = get_anchor_target_for_item(category, super_category, anchors)
        if target:
            return yaw_to_face_target(x_norm, z_norm, target["x_norm"], target["z_norm"])
        if zone_l == "seat_zone":
            return 180.0
        return 0.0

    if "table" in category_l or super_l == "table":
        return 0.0

    return 0.0


# =========================================================
# DATA HELPERS
# =========================================================
VALID_SUPER_CATEGORIES = {
    "Bed",
    "Sofa",
    "Table",
    "Cabinet/Shelf/Desk",
    "Chair",
    "Lighting",
    "Pier/Stool",
    "Appliance",
    "Plumbing",
    "Other",
}


def row_matches_room_type(row, room_type: str) -> bool:
    row_room_type = safe_str(row.get("room_type"), "")
    if row_room_type == room_type:
        return True

    room_types_raw = safe_str(row.get("room_types"), "")
    if room_types_raw:
        room_types = [x.strip() for x in room_types_raw.split("|") if x.strip()]
        if room_type in room_types:
            return True

    return False


def clean_dataset(df):
    df = df.copy()

    required_cols = ["model_id", "super-category", "category"]
    existing_required = [c for c in required_cols if c in df.columns]

    if existing_required:
        df = df.dropna(subset=existing_required)

    for col in ["super-category", "category", "room_type", "room_types"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "super-category" in df.columns:
        df = df[df["super-category"].isin(VALID_SUPER_CATEGORIES)]

    if "category" in df.columns:
        broken_lighting = ["lighting", "light", "lamp"]
        df = df[~df["category"].str.lower().isin(broken_lighting)]

    return df


# =========================================================
# TRUE MULTIROOM COMPATIBILITY
# =========================================================
def room_compatible_score(row, room_type):
    score = 0

    row_room_type = safe_str(row.get("room_type"), "")
    if row_room_type == room_type:
        score += 10

    room_types_raw = safe_str(row.get("room_types"), "")
    room_types = [r.strip() for r in room_types_raw.split("|") if r.strip()]
    if room_type in room_types:
        score += 8

    if safe_float(row.get("is_multiroom"), 0) > 0:
        score += 3

    room_count = safe_float(row.get("room_count"), 1)
    score += min(room_count, 5)

    return score


def get_recently_used_model_ids(limit_predictions=20):
    used_ids = set()

    try:
        docs = (
            prediction_collection
            .order_by("timestamp_completed_unix", direction=firestore.Query.DESCENDING)
            .limit(limit_predictions)
            .stream()
        )

        for doc in docs:
            data = doc.to_dict() or {}
            prediction_results = data.get("prediction_results", {})
            furniture = prediction_results.get("furniture", [])

            for item in furniture:
                model_id = str(item.get("model_id", "")).strip()
                if model_id:
                    used_ids.add(model_id)

    except Exception as e:
        print(f"[History] Could not load recent model ids: {e}")

    return used_ids


def get_furniture_counts(room_type):
    return {
        "Bedroom": {
            "Bed": 1,
            "Cabinet/Shelf/Desk": 1,
            "Chair": 1,
            "Lighting": 1,
        },
        "Living Room": {
            "Sofa": 1,
            "Table": 1,
            "Cabinet/Shelf/Desk": 1,
            "Chair": 1,
            "Lighting": 1,
        },
        "Dining Room": {
            "Table": 1,
            "Chair": 4,
            "Cabinet/Shelf/Desk": 1,
            "Lighting": 1,
        },
        "Kitchen Room": {
            "Appliance": 3,
            "Plumbing": 1,
            "Cabinet/Shelf/Desk": 1,
            "Lighting": 1,
        },
        "Bathroom": {
            "Plumbing": 2,
            "Appliance": 1,
            "Cabinet/Shelf/Desk": 1,
            "Lighting": 1,
        },
    }.get(room_type, {})


ROOM_CATEGORY_RULES = {
    "Bedroom": {
        "Bed": {
            "categories": [
                "King-size Bed", "Single bed", "Bed Frame",
                "Bunk Bed", "Kids Bed",
            ],
            "priority": 10,
        },
        "Cabinet/Shelf/Desk": {
            "categories": [
                "Nightstand", "Wardrobe", "Children Cabinet",
                "Drawer Chest / Corner cabinet", "Dressing Table",
                "Desk", "Shelf", "Bookcase / jewelry Armoire", "Shoe Cabinet",
            ],
            "priority": 7,
        },
        "Chair": {
            "categories": [
                "Dressing Chair", "Classic Chinese Chair",
                "Lounge Chair / Cafe Chair / Office Chair",
                "Footstool / Sofastool / Bed End Stool / Stool",
                "Folding chair",
            ],
            "priority": 5,
        },
        "Lighting": {
            "categories": ["Ceiling Lamp", "Pendant Lamp", "Wall Lamp", "Floor Lamp"],
            "priority": 6,
        },
    },
    "Living Room": {
        "Sofa": {
            "categories": [
                "Three-Seat / Multi-seat Sofa", "Loveseat Sofa", "L-shaped Sofa",
                "U-shaped Sofa", "Couch Bed", "armchair", "Lazy Sofa",
                "Chaise Longue Sofa",
            ],
            "priority": 10,
        },
        "Table": {
            "categories": ["Coffee Table", "Corner/Side Table", "Round End Table"],
            "priority": 7,
        },
        "Cabinet/Shelf/Desk": {
            "categories": [
                "TV Stand", "Shelf", "Bookcase / jewelry Armoire",
                "Sideboard / Side Cabinet / Console Table", "Wine Cabinet", "Desk",
            ],
            "priority": 6,
        },
        "Chair": {
            "categories": [
                "Lounge Chair / Cafe Chair / Office Chair", "Hanging Chair",
                "Folding chair", "Footstool / Sofastool / Bed End Stool / Stool",
            ],
            "priority": 5,
        },
        "Lighting": {
            "categories": ["Ceiling Lamp", "Pendant Lamp", "Wall Lamp", "Floor Lamp"],
            "priority": 6,
        },
    },
    "Dining Room": {
        "Table": {
            "categories": ["Dining Table", "Bar"],
            "priority": 10,
        },
        "Chair": {
            "categories": ["Dining Chair", "Barstool", "Folding chair"],
            "priority": 9,
        },
        "Cabinet/Shelf/Desk": {
            "categories": [
                "Sideboard / Side Cabinet / Console Table", "Wine Cabinet", "Shelf",
            ],
            "priority": 5,
        },
        "Lighting": {
            "categories": ["Ceiling Lamp", "Pendant Lamp", "Wall Lamp"],
            "priority": 6,
        },
    },
    "Kitchen Room": {
        "Cabinet/Shelf/Desk": {
            "categories": ["Wine Cabinet", "Shelf"],
            "priority": 5,
        },
        "Lighting": {
            "categories": ["Ceiling Lamp", "Pendant Lamp", "Wall Lamp"],
            "priority": 6,
        },
    },
    "Bathroom": {
        "Lighting": {
            "categories": ["Ceiling Lamp", "Wall Lamp"],
            "priority": 6,
        },
    },
}


def select_items_for_room(df_filtered, room_type, recent_used_ids=None):
    if recent_used_ids is None:
        recent_used_ids = set()

    furniture_counts = get_furniture_counts(room_type)
    all_items = []

    for super_category, count in furniture_counts.items():
        candidates = df_filtered[
            df_filtered["super-category"] == super_category
        ].copy()

        if candidates.empty:
            continue

        candidates = candidates[
            candidates.apply(lambda r: row_matches_room_type(r, room_type), axis=1)
        ].copy()

        if candidates.empty:
            continue

        candidates["room_score"] = candidates.apply(
            lambda r: room_compatible_score(r, room_type), axis=1
        )

        room_rules = (
            ROOM_CATEGORY_RULES
            .get(room_type, {})
            .get(super_category)
        )

        if room_rules:
            allowed_categories = room_rules.get("categories", [])
            priority = room_rules.get("priority", 1)

            if allowed_categories:
                candidates = candidates[
                    candidates["category"].isin(allowed_categories)
                ].copy()

                candidates["spawn_weight"] = priority + candidates["room_score"]

                if "is_multiroom" in candidates.columns:
                    candidates["spawn_weight"] = (
                        candidates["spawn_weight"]
                        + candidates["is_multiroom"].fillna(0) * 2
                    )

        if candidates.empty:
            continue

        candidates = candidates.drop_duplicates(subset=["model_id"])

        fresh_candidates = candidates[
            ~candidates["model_id"].astype(str).isin(recent_used_ids)
        ].copy()

        pool = fresh_candidates if len(fresh_candidates) >= count else candidates

        if "room_score" in pool.columns:
            pool = pool.sort_values(by="room_score", ascending=False)

        if "spawn_weight" in pool.columns:
            pool = pool.sample(
                frac=1.0,
                weights=pool["spawn_weight"],
                random_state=random.randint(0, 999999)
            )
        else:
            pool = pool.sample(frac=1.0, random_state=random.randint(0, 999999))

        picked = []
        used_local = set()

        for _, row in pool.iterrows():
            model_id = str(row["model_id"]).strip()
            if model_id in used_local:
                continue
            picked.append(row)
            used_local.add(model_id)
            if len(picked) >= count:
                break

        all_items.extend(picked)

    return all_items


def category_priority(category, super_category):
    if category in ["King-size Bed", "Bunk Bed", "Bed Frame", "Single bed", "Kids Bed"]:
        return 1
    if category in ["Three-Seat / Multi-seat Sofa", "Loveseat Sofa", "L-shaped Sofa", "U-shaped Sofa", "Couch Bed", "armchair", "Lazy Sofa", "Chaise Longue Sofa"]:
        return 2
    if category in ["Dining Table", "Desk", "Dressing Table", "Bar", "Coffee Table"]:
        return 3
    if category in ["Wardrobe", "TV Stand", "Shelf", "Bookcase / jewelry Armoire", "Children Cabinet", "Drawer Chest / Corner cabinet", "Wine Cabinet", "Sideboard / Side Cabinet / Console Table", "Shoe Cabinet"]:
        return 4
    if category in ["Nightstand", "Corner/Side Table", "Round End Table"]:
        return 5
    if super_category == "Chair":
        return 6
    if super_category == "Lighting":
        return 7
    return 10


def category_defaults(df_ref, category, room_type=None):
    subset = df_ref[df_ref["category"] == category].copy()

    if room_type is not None and not subset.empty:
        subset = subset[subset.apply(lambda r: row_matches_room_type(r, room_type), axis=1)].copy()

    if subset.empty:
        return {
            "style": "unknown",
            "theme": "unknown",
            "material": "unknown",
            "super-category": "Other",
            "anchor_type": "secondary",
            "placement_zone": "general",
            "fov": 0.785398,
        }

    first = subset.iloc[0]
    return {
        "style": safe_str(first.get("style"), "unknown"),
        "theme": safe_str(first.get("theme"), "unknown"),
        "material": safe_str(first.get("material"), "unknown"),
        "super-category": safe_str(first.get("super-category"), "Other"),
        "anchor_type": safe_str(first.get("anchor_type"), "secondary"),
        "placement_zone": safe_str(first.get("placement_zone"), "general"),
        "fov": safe_float(first.get("fov"), 0.785398),
    }


def build_model_features(row, room_type, room_width, room_length, room_height):
    category = safe_str(row.get("category"))
    defaults = category_defaults(df_master, category, room_type=room_type)

    room_types_raw = safe_str(row.get("room_types"), "")
    room_count = len([r for r in room_types_raw.split("|") if r.strip()])
    is_multiroom = 1 if room_count > 1 else 0

    feature_row = {
        "super-category": safe_str(row.get("super-category", defaults["super-category"])),
        "category": category,
        "style": safe_str(row.get("style"), defaults["style"]),
        "theme": safe_str(row.get("theme"), defaults["theme"]),
        "material": safe_str(row.get("material"), defaults["material"]),
        "room_type": safe_str(room_type),
        "anchor_type": safe_str(row.get("anchor_type"), defaults["anchor_type"]),
        "placement_zone": safe_str(row.get("placement_zone"), defaults["placement_zone"]),
        "source_dataset": safe_str(row.get("source_dataset"), "unknown"),
        "room_width": safe_float(room_width, 4.0),
        "room_length": safe_float(room_length, 5.0),
        "room_height": safe_float(room_height, 2.8),
        "room_area": safe_float(room_width, 4.0) * safe_float(room_length, 5.0),
        "is_multiroom": is_multiroom,
        "room_count": room_count,
    }

    for rank in [1, 2, 3]:
        feature_row[f"recommendation_{rank}_shop"] = safe_str(row.get(f"recommendation_{rank}_shop"), "unknown")
        feature_row[f"recommendation_{rank}_availability"] = safe_str(row.get(f"recommendation_{rank}_availability"), "unknown")
        feature_row[f"recommendation_{rank}_physical_store_place"] = safe_str(row.get(f"recommendation_{rank}_physical_store_place"), "unknown")
        feature_row[f"recommendation_{rank}_source"] = safe_str(row.get(f"recommendation_{rank}_source"), "unknown")

    return pd.DataFrame([feature_row])

def predict_ai_depth(row, room_type, room_width, room_length, room_height):
    X = build_model_features(row, room_type, room_width, room_length, room_height)
    return clamp01(depth_model.predict(X)[0])


def predict_ai_scale(row, room_type, room_width, room_length, room_height):
    X = build_model_features(row, room_type, room_width, room_length, room_height)
    return clamp(scale_model.predict(X)[0], 0.5, 1.8)


def apply_smart_metadata(pred):
    metadata = load_smart_metadata(pred["model_id"])

    if metadata is None:
        return pred

    pred["x_norm"] = clamp01(pred["x_norm"] + safe_float(metadata.get("position_offset_x", 0.0)))
    pred["y_norm"] = clamp01(pred["y_norm"] + safe_float(metadata.get("position_offset_y", 0.0)))
    pred["z_norm"] = clamp01(pred["z_norm"] + safe_float(metadata.get("position_offset_z", 0.0)))

    # FIX: apply smart offsets to x_norm/z_norm FIRST, then recompute x/z metres
    # so that x == x_norm * room_width is always true. Previously x was offset
    # in metres but x_norm was not updated, leaving them inconsistent.
    # Unity now reads x_norm/z_norm, so consistency here is critical.

    pred["rotation_x"] += safe_float(metadata.get("rotation_offset_x", 0.0))
    pred["rotation_y"] += safe_float(metadata.get("rotation_offset_y", 0.0))
    pred["rotation_z"] += safe_float(metadata.get("rotation_offset_z", 0.0))
    pred["rotation_y"] = snap_rotation_90(pred["rotation_y"])

    pred["scale_x"] *= safe_float(metadata.get("scale_multiplier_x", 1.0))
    pred["scale_y"] *= safe_float(metadata.get("scale_multiplier_y", 1.0))
    pred["scale_z"] *= safe_float(metadata.get("scale_multiplier_z", 1.0))

    print(f"[SMART] applied metadata to {pred['model_id']}")
    return pred


# =========================================================
# UNIVERSAL SPATIAL RULES
# =========================================================
SPATIAL_RULES = {
    "King-size Bed": {
        "placement": "against_wall",
        "preferred_wall": "back_wall",
        "clearance_front": 0.25,
        "support_surface": "floor",
    },
    "Single bed": {
        "placement": "against_wall",
        "preferred_wall": "back_wall",
        "clearance_front": 0.25,
        "support_surface": "floor",
    },
    "Dining Chair": {
        "placement": "near_anchor",
        "anchor_types": ["Dining Table"],
        "distance": 0.12,
        "face_anchor": True,
        "support_surface": "floor",
    },
    "Lounge Chair / Cafe Chair / Office Chair": {
        "placement": "near_anchor",
        "anchor_types": ["Desk", "Coffee Table"],
        "distance": 0.12,
        "face_anchor": True,
        "support_surface": "floor",
    },
    "Dining Table": {
        "placement": "center",
        "support_surface": "floor",
    },
    "Coffee Table": {
        "placement": "front_of_anchor",
        "anchor_types": ["Three-Seat / Multi-seat Sofa"],
        "distance": 0.18,
        "support_surface": "floor",
    },
    "Pendant Lamp": {
        "placement": "above_anchor",
        "anchor_types": ["King-size Bed", "Dining Table", "Coffee Table"],
        "support_surface": "ceiling",
        "allow_intersection": False,
        "height_offset": 0.25,
    },
    "Ceiling Lamp": {
        "placement": "ceiling_center",
        "support_surface": "ceiling",
    },
    "Wall Lamp": {
        "placement": "wall_attached",
        "support_surface": "wall",
    },
    "Floor Lamp": {
        "placement": "corner",
        "support_surface": "floor",
    },
    "Wardrobe": {
        "placement": "against_wall",
        "preferred_wall": "side_wall",
        "support_surface": "floor",
    },
    "TV Stand": {
        "placement": "opposite_anchor",
        "anchor_types": ["Three-Seat / Multi-seat Sofa"],
        "face_anchor": True,
        "support_surface": "floor",
    },
}

OBJECT_BEHAVIOR = {
    "Pendant Lamp": "hanging",
    "Ceiling Lamp": "ceiling",
    "Wall Lamp": "wall_mounted",
    "Floor Lamp": "floor_supported",
    "Dining Chair": "floor_supported",
    "Lounge Chair / Cafe Chair / Office Chair": "floor_supported",
    "King-size Bed": "large_floor_object",
    "Wardrobe": "wall_aligned",
    "TV Stand": "wall_facing",
}

OBJECT_FOOTPRINTS = {
    "King-size Bed":   {"width": 0.32, "depth": 0.42},
    "Single bed":      {"width": 0.22, "depth": 0.36},
    "Three-Seat / Multi-seat Sofa": {"width": 0.32, "depth": 0.18},
    "Loveseat Sofa":   {"width": 0.24, "depth": 0.16},
    "Dining Table":    {"width": 0.28, "depth": 0.28},
    "Coffee Table":    {"width": 0.18, "depth": 0.14},
    "Dining Chair":    {"width": 0.12, "depth": 0.12},
    "Lounge Chair / Cafe Chair / Office Chair": {"width": 0.14, "depth": 0.14},
    "Wardrobe":        {"width": 0.18, "depth": 0.12},
    "TV Stand":        {"width": 0.24, "depth": 0.10},
    "Floor Lamp":      {"width": 0.08, "depth": 0.08},
    "Pendant Lamp":    {"width": 0.10, "depth": 0.10},
}


# =========================================================
# TEMPLATE LAYOUT
# =========================================================
def get_anchor_registry():
    return {"bed": None, "sofa": None, "table": None, "desk": None, "tv": None, "wardrobe": None}


def store_anchor(anchor_registry, obj):
    category = obj["category"]
    record = {
        "x_norm": obj["x_norm"],
        "z_norm": obj["z_norm"],
        "x": obj["x"],
        "z": obj["z"],
        "rotation_y": obj["rotation_y"],
        "category": category,
    }

    if category in ["King-size Bed", "Bunk Bed", "Bed Frame", "Single bed", "Kids Bed"]:
        anchor_registry["bed"] = record
    elif category in ["Three-Seat / Multi-seat Sofa", "Loveseat Sofa", "L-shaped Sofa", "U-shaped Sofa", "Couch Bed", "armchair", "Lazy Sofa", "Chaise Longue Sofa"]:
        anchor_registry["sofa"] = record
    elif category in ["Dining Table", "Bar", "Coffee Table"]:
        anchor_registry["table"] = record
    elif category in ["Desk", "Dressing Table"]:
        anchor_registry["desk"] = record
    elif category == "TV Stand":
        anchor_registry["tv"] = record
    elif category in ["Wardrobe", "Children Cabinet", "Drawer Chest / Corner cabinet", "Shelf", "Bookcase / jewelry Armoire", "Wine Cabinet", "Sideboard / Side Cabinet / Console Table", "Shoe Cabinet"]:
        anchor_registry["wardrobe"] = record


def room_scale_x(x_norm, room_width):
    return clamp(x_norm, 0.05, 0.95) * room_width


def room_scale_z(z_norm, room_length):
    return clamp(z_norm, 0.05, 0.95) * room_length


# =========================================================
# UNIVERSAL SPATIAL PLACEMENT ENGINE
# FIX: indentation bugs corrected — window/door avoidance
# blocks were not inside the function body properly.
# =========================================================
def generate_spatial_layout(item, anchors, scene_meta, room_type):
    category = safe_str(item.get("category"))
    rule = SPATIAL_RULES.get(category, {})
    placement = rule.get("placement", "general")

    center_x = scene_meta.get(
        "usable_center_x",
        scene_meta.get("back_wall_center_x", 0.50)
    )

    x_norm = center_x
    z_norm = 0.50
    rotation_y = 0.0

    if placement == "against_wall":
        preferred_wall = rule.get("preferred_wall", "back_wall")

        if preferred_wall == "back_wall":
            x_norm = center_x
            z_norm = 0.18
            rotation_y = 180.0

            # Window safety — shift bed away from window centre
            if 0.42 <= x_norm <= 0.58:
                x_norm += random.choice([-0.15, 0.15])

        elif preferred_wall == "side_wall":
            x_norm = 0.15
            z_norm = 0.45
            rotation_y = 90.0

    elif placement == "center":
        x_norm = center_x
        z_norm = 0.50
        rotation_y = 0.0

    elif placement == "above_anchor":
        anchor_types = rule.get("anchor_types", [])
        target_anchor = None

        for anchor_name in anchor_types:
            for key, value in anchors.items():
                if value and value.get("category") == anchor_name:
                    target_anchor = value
                    break

        if target_anchor:
            x_norm = target_anchor["x_norm"]
            z_norm = max(0.10, target_anchor["z_norm"] - 0.08)
        else:
            x_norm = center_x
            z_norm = 0.22

        rotation_y = 0.0

    elif placement == "front_of_anchor":
        sofa = anchors.get("sofa")
        if sofa:
            x_norm = sofa["x_norm"]
            z_norm = sofa["z_norm"] + 0.18
        else:
            x_norm = center_x
            z_norm = 0.42
        rotation_y = 0.0

    elif placement == "opposite_anchor":
        sofa = anchors.get("sofa")
        if sofa:
            x_norm = sofa["x_norm"]
            z_norm = 0.82
        else:
            x_norm = center_x
            z_norm = 0.80
        rotation_y = 0.0

    elif placement == "corner":
        x_norm = 0.82
        z_norm = 0.22
        rotation_y = 180.0

    elif placement == "wall_attached":
        wall_lines = scene_meta.get("wall_lines", [])
        if wall_lines:
            best_wall = wall_lines[0]
            image_w = 1000
            wall_center_x = ((best_wall["x1"] + best_wall["x2"]) / 2) / image_w
            x_norm = clamp01(wall_center_x)
        else:
            x_norm = 0.14
        z_norm = 0.35
        rotation_y = 90.0

    elif placement == "ceiling_center":
        x_norm = center_x
        z_norm = 0.30
        rotation_y = 0.0

    # FIX: walkway, window, and door avoidance are now correctly
    # indented inside the function (were at module level before).
    if room_type in ["Bedroom", "Living Room"]:
        if 0.40 <= x_norm <= 0.60 and 0.40 <= z_norm <= 0.70:
            x_norm += random.choice([-0.12, 0.12])

        windows = scene_meta.get("windows", [])
        for win in windows:
            wx = safe_float(win.get("x_norm"), 0.5)
            if abs(x_norm - wx) < 0.12:
                if x_norm <= wx:
                    x_norm -= 0.15
                else:
                    x_norm += 0.15

    doors = scene_meta.get("doors", [])
    for door in doors:
        dx = safe_float(door.get("x_norm"), 0.5)
        if abs(x_norm - dx) < 0.15:
            z_norm += 0.12

    return (clamp01(x_norm), clamp01(z_norm), rotation_y)


def bedroom_template(item, relation_index, anchors, ai_z_norm, width, length, scene_meta):
    category = item["category"]
    wide_room = width >= 5.0
    long_room = length >= 6.0

    center_x = scene_meta.get("back_wall_center_x", 0.50)
    left_x = scene_meta.get("usable_left_x", 0.12)
    right_x = scene_meta.get("usable_right_x", 0.88)

    x_norm = center_x
    z_norm = ai_z_norm
    rotation_y = 180.0

    if category in ["King-size Bed", "Bunk Bed", "Bed Frame", "Single bed", "Kids Bed"]:
        x_norm = center_x
        z_norm = 0.16 if long_room else 0.20
        rotation_y = 180.0

    elif category == "Nightstand":
        bed = anchors["bed"]
        if bed:
            x_norm = max(left_x + 0.05, bed["x_norm"] - 0.12) if relation_index % 2 == 0 else min(right_x - 0.05, bed["x_norm"] + 0.12)
            z_norm = min(0.28, bed["z_norm"] + 0.07)
        else:
            x_norm = min(right_x - 0.05, center_x + 0.12)
            z_norm = 0.27
        rotation_y = 180.0

    elif category in ["Wardrobe", "Children Cabinet", "Drawer Chest / Corner cabinet", "Shelf", "Bookcase / jewelry Armoire", "Shoe Cabinet"]:
        x_norm = left_x if relation_index % 2 == 0 else right_x
        z_norm = 0.40 if wide_room else 0.45
        rotation_y = 90.0 if relation_index % 2 == 0 else 270.0

    elif category in ["Desk", "Dressing Table"]:
        x_norm = min(right_x, max(left_x, left_x + 0.08))
        z_norm = 0.58
        rotation_y = 90.0

    elif item["super-category"] == "Chair":
        desk = anchors["desk"]
        if desk:
            x_norm = desk["x_norm"]
            z_norm = min(0.80, desk["z_norm"] + 0.08)
        else:
            x_norm = right_x - 0.08
            z_norm = 0.64
        rotation_y = 180.0

    elif category in ["Ceiling Lamp", "Pendant Lamp"]:
        bed = anchors["bed"]
        if bed:
            x_norm = bed["x_norm"]
            z_norm = bed["z_norm"]
        else:
            x_norm = center_x
            z_norm = 0.30
        rotation_y = 0.0

    return x_norm, z_norm, rotation_y


def living_room_template(item, relation_index, anchors, ai_z_norm, width, length, scene_meta):
    category = item["category"]
    wide_room = width >= 5.5
    long_room = length >= 6.5

    center_x = scene_meta.get("back_wall_center_x", 0.50)
    left_x = scene_meta.get("usable_left_x", 0.12)
    right_x = scene_meta.get("usable_right_x", 0.88)

    x_norm = center_x
    z_norm = ai_z_norm
    rotation_y = 180.0

    if category in ["Three-Seat / Multi-seat Sofa", "Loveseat Sofa", "L-shaped Sofa", "U-shaped Sofa", "Couch Bed", "armchair", "Lazy Sofa", "Chaise Longue Sofa"]:
        x_norm = center_x
        z_norm = 0.18 if long_room else 0.22
        rotation_y = 180.0

    elif category == "TV Stand":
        sofa = anchors["sofa"]
        x_norm = sofa["x_norm"] if sofa else center_x
        z_norm = 0.80
        rotation_y = 0.0

    elif category == "Coffee Table":
        sofa = anchors["sofa"]
        x_norm = sofa["x_norm"] if sofa else center_x
        z_norm = 0.42
        rotation_y = 0.0

    elif item["super-category"] == "Chair":
        x_norm = left_x + 0.08 if relation_index % 2 == 0 else right_x - 0.08
        z_norm = 0.45
        rotation_y = 180.0

    elif category in ["Corner/Side Table", "Round End Table"]:
        x_norm = left_x + 0.10 if relation_index % 2 == 0 else right_x - 0.10
        z_norm = 0.30
        rotation_y = 180.0

    elif category in ["Shelf", "Bookcase / jewelry Armoire", "Children Cabinet", "Wine Cabinet", "Sideboard / Side Cabinet / Console Table", "Shoe Cabinet"]:
        x_norm = left_x if relation_index % 2 == 0 else right_x
        z_norm = 0.46 if wide_room else 0.50
        rotation_y = 90.0 if relation_index % 2 == 0 else 270.0

    elif category in ["Ceiling Lamp", "Pendant Lamp", "Floor Lamp", "Wall Lamp"]:
        sofa = anchors["sofa"]
        x_norm = sofa["x_norm"] if sofa else center_x
        z_norm = 0.44 if sofa else 0.35
        rotation_y = 0.0

    return x_norm, z_norm, rotation_y


def dining_room_template(item, relation_index, anchors, ai_z_norm, width, length, scene_meta):
    category = item["category"]

    center_x = scene_meta.get("back_wall_center_x", 0.50)
    left_x = scene_meta.get("usable_left_x", 0.12)
    right_x = scene_meta.get("usable_right_x", 0.88)

    x_norm = center_x
    z_norm = ai_z_norm
    rotation_y = 0.0

    if category in ["Dining Table", "Bar"]:
        x_norm = center_x
        z_norm = 0.50
        rotation_y = 0.0

    elif item["super-category"] == "Chair":
        chair_positions = [
            (center_x, 0.35, 0.0),
            (center_x, 0.65, 180.0),
            (max(left_x + 0.08, center_x - 0.17), 0.50, 90.0),
            (min(right_x - 0.08, center_x + 0.17), 0.50, 270.0),
        ]
        x_norm, z_norm, rotation_y = chair_positions[relation_index % len(chair_positions)]

    elif category in ["Sideboard / Side Cabinet / Console Table", "Shelf", "Wine Cabinet"]:
        x_norm = left_x if relation_index % 2 == 0 else right_x
        z_norm = 0.48
        rotation_y = 90.0 if relation_index % 2 == 0 else 270.0

    elif category in ["Ceiling Lamp", "Pendant Lamp"]:
        table = anchors["table"]
        x_norm = table["x_norm"] if table else center_x
        z_norm = table["z_norm"] if table else 0.50
        rotation_y = 0.0

    return x_norm, z_norm, rotation_y


def apply_template(room_type, item, relation_index, anchors, ai_z_norm, width, length, scene_meta):
    if room_type == "Bedroom":
        return bedroom_template(item, relation_index, anchors, ai_z_norm, width, length, scene_meta)
    if room_type == "Living Room":
        return living_room_template(item, relation_index, anchors, ai_z_norm, width, length, scene_meta)
    if room_type == "Dining Room":
        return dining_room_template(item, relation_index, anchors, ai_z_norm, width, length, scene_meta)
    return 0.50, ai_z_norm, 180.0


# =========================================================
# FOOTPRINT LOOKUP
# =========================================================
def get_object_footprint(category):
    fp = OBJECT_FOOTPRINTS.get(category, None)
    if fp is None:
        return {"width": 0.14, "depth": 0.14}
    return fp


# =========================================================
# REAL BOUNDING BOX COLLISION
# =========================================================
def normalized_collision_adjust(current, placed, room_type):
    x_norm = current["x_norm"]
    z_norm = current["z_norm"]

    current_fp = get_object_footprint(current["category"])
    current_w = current_fp["width"]
    current_d = current_fp["depth"]

    walkway_padding = 0.10

    for obj in placed:
        obj_fp = get_object_footprint(obj["category"])
        obj_w = obj_fp["width"]
        obj_d = obj_fp["depth"]

        dx = abs(x_norm - obj["x_norm"])
        dz = abs(z_norm - obj["z_norm"])

        overlap_x = dx < ((current_w / 2) + (obj_w / 2) + walkway_padding)
        overlap_z = dz < ((current_d / 2) + (obj_d / 2) + walkway_padding)

        if overlap_x and overlap_z:
            push_x = (current_w / 2) + (obj_w / 2) + walkway_padding
            push_z = (current_d / 2) + (obj_d / 2) + walkway_padding

            if dx < dz:
                if x_norm <= obj["x_norm"]:
                    x_norm -= push_x
                else:
                    x_norm += push_x
            else:
                if z_norm <= obj["z_norm"]:
                    z_norm -= push_z
                else:
                    z_norm += push_z

    x_norm = clamp(x_norm, 0.05, 0.95)
    z_norm = clamp(z_norm, 0.05, 0.95)

    current["x_norm"] = x_norm
    current["z_norm"] = z_norm

    return current


def apply_layout_scale(role, ai_scale, width, length, height):
    room_area = width * length
    area_factor = clamp(room_area / 25.0, 0.8, 1.2)
    height_factor = clamp(height / 2.8, 0.85, 1.2)
    final_factor = area_factor * height_factor

    if role == "bed_anchor":
        return clamp(ai_scale * final_factor, 0.7, 1.6)
    if role == "sofa_anchor":
        return clamp(ai_scale * final_factor, 0.7, 1.6)
    if role == "table_anchor":
        return clamp(ai_scale * final_factor, 0.7, 1.5)
    if role == "storage":
        return clamp(ai_scale * 0.95 * final_factor, 0.7, 1.6)
    if role == "seat":
        return clamp(ai_scale * 0.90 * final_factor, 0.6, 1.3)
    if role == "light":
        return clamp(ai_scale * height_factor, 0.5, 1.2)
    return clamp(ai_scale * final_factor, 0.6, 1.4)


# =========================================================
# KITCHEN / BATHROOM RULE-BASED GENERATOR
# =========================================================
def normalize_room_value(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def supports_kb_room(row, room_type: str) -> bool:
    room_aliases = {
        "Kitchen": "Kitchen Room",
        "Kitchen Room": "Kitchen Room",
        "Bathroom": "Bathroom",
        "Bath": "Bathroom",
        "Toilet": "Bathroom",
    }
    room_type = room_aliases.get(str(room_type).strip(), str(room_type).strip())

    room_value = normalize_room_value(row.get("room_type", ""))
    usable_value = normalize_room_value(row.get("usable_room_types", ""))

    combined = "|".join([room_value, usable_value])
    room_types = [x.strip() for x in combined.split("|") if x.strip()]

    normalized_room_types = [room_aliases.get(x, x) for x in room_types]
    return room_type in normalized_room_types


def clean_kb_category(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def get_kb_category(row):
    category_clean = clean_kb_category(row.get("category_clean", ""))
    if category_clean:
        return category_clean

    category = clean_kb_category(row.get("category", ""))
    if "," in category:
        return category.split(",")[0].strip()

    return category


def get_kb_object_rules(room_type):
    if room_type in ["Kitchen Room", "Kitchen"]:
        return {
            "stove": 1,
            "microwave": 1,
            "washer": 1,
            "faucet": 1,
            "water faucet": 1,
            "mixing faucet": 1,
        }
    if room_type == "Bathroom":
        return {
            "bathtub": 1,
            "washer": 1,
            "faucet": 1,
            "water faucet": 1,
            "mixing faucet": 1,
        }
    return {}


def select_kitchen_bathroom_items(room_type, recent_used_ids=None, style=None):
    if recent_used_ids is None:
        recent_used_ids = set()

    if kb_df.empty:
        return []

    object_rules = get_kb_object_rules(room_type)
    selected = []

    kb_work = kb_df.copy()
    kb_work["category_clean_runtime"] = kb_work.apply(get_kb_category, axis=1)

    style_requested = safe_str(style, "").strip()
    if style_requested and "style" in kb_work.columns:
        kb_style_filtered = kb_work[kb_work["style"].astype(str) == style_requested].copy()
        if not kb_style_filtered.empty:
            kb_work = kb_style_filtered
            print(f"[Kitchen/Bathroom] style filter applied: {style_requested}")
        else:
            print(f"[Kitchen/Bathroom] no exact style match for {style_requested}; using all styles")

    for category_clean, count in object_rules.items():
        candidates = kb_work[
            (kb_work["category_clean_runtime"] == category_clean)
            & (kb_work.apply(lambda r: supports_kb_room(r, room_type), axis=1))
        ].copy()

        if candidates.empty:
            continue

        candidates = candidates.drop_duplicates(subset=["model_id"])

        fresh_candidates = candidates[
            ~candidates["model_id"].astype(str).isin(recent_used_ids)
        ].copy()

        pool = fresh_candidates if len(fresh_candidates) >= count else candidates
        pool = pool.sample(frac=1.0, random_state=random.randint(0, 999999))

        picked = []
        used_local = set()

        for _, row in pool.iterrows():
            model_id = str(row["model_id"]).strip()
            if model_id in used_local:
                continue
            picked.append(row)
            used_local.add(model_id)
            if len(picked) >= count:
                break

        selected.extend(picked)

    return selected


def kb_rule_position(room_type, category_clean, zone):
    category_clean = clean_kb_category(category_clean)
    zone = clean_kb_category(zone)

    # =====================================================
    # KITCHEN
    # =====================================================

    if room_type in ["Kitchen Room", "Kitchen"]:

        # BACK COUNTER ZONE
        if category_clean in [
            "stove",
            "microwave",
            "water faucet",
            "mixing faucet",
            "faucet"
        ]:
            x_norm = random.uniform(0.25, 0.75)
            z_norm = random.uniform(0.08, 0.18)
            return x_norm, z_norm, 0.0, 180.0, "counter_wall"

        # SIDE APPLIANCE ZONE
        if category_clean in [
            "washer"
        ]:
            side = random.choice(["left", "right"])

            if side == "left":
                x_norm = random.uniform(0.08, 0.18)
                rotation = 90.0
            else:
                x_norm = random.uniform(0.82, 0.92)
                rotation = 270.0

            z_norm = random.uniform(0.25, 0.75)

            return x_norm, z_norm, 0.0, rotation, "side_wall"

    # =====================================================
    # BATHROOM
    # =====================================================

    if room_type == "Bathroom":

        # BATHTUB BACK WALL
        if category_clean == "bathtub":
            return (
                random.uniform(0.30, 0.70),
                random.uniform(0.08, 0.15),
                0.0,
                180.0,
                "back_wall"
            )

        # TOILET SIDE WALL
        if "toilet" in category_clean:

            side = random.choice(["left", "right"])

            if side == "left":
                return (
                    random.uniform(0.08, 0.18),
                    random.uniform(0.30, 0.65),
                    0.0,
                    90.0,
                    "side_wall"
                )

            return (
                random.uniform(0.82, 0.92),
                random.uniform(0.30, 0.65),
                0.0,
                270.0,
                "side_wall"
            )

        # SINK AREA
        if category_clean in [
            "faucet",
            "water faucet",
            "mixing faucet",
            "sink"
        ]:

            return (
                random.uniform(0.25, 0.75),
                random.uniform(0.12, 0.22),
                0.55,
                180.0,
                "sink_zone"
            )

        # WASHER
        if category_clean == "washer":

            return (
                random.uniform(0.75, 0.90),
                random.uniform(0.70, 0.88),
                0.0,
                0.0,
                "service_corner"
            )

    # =====================================================
    # FALLBACK
    # =====================================================

    return (
        random.uniform(0.20, 0.80),
        random.uniform(0.20, 0.80),
        0.0,
        random.choice([0.0, 90.0, 180.0, 270.0]),
        "general"
    )


def kb_collision_adjust(current, placed):
    x_norm = current["x_norm"]
    z_norm = current["z_norm"]

    min_dx = 0.08
    min_dz = 0.08

    for obj in placed:
        dx = abs(x_norm - obj["x_norm"])
        dz = abs(z_norm - obj["z_norm"])

        if dx < min_dx and dz < min_dz:
            if current["zone"] in ["sink_zone", "counter", "counter_wall"]:
                x_norm = min(0.95, x_norm + 0.10)
            else:
                z_norm = min(0.90, z_norm + 0.10)

    current["x_norm"] = clamp01(x_norm)
    current["z_norm"] = clamp01(z_norm)
    return current


def map_kb_super_category(category_clean, role):
    category_clean = clean_kb_category(category_clean)
    role = clean_kb_category(role)

    if category_clean in ["stove", "microwave", "washer"] or role == "appliance":
        return "Appliance"
    if category_clean in ["bathtub", "faucet", "water faucet", "mixing faucet"] or role == "fixture":
        return "Plumbing"
    return "Other"


def generate_kitchen_bathroom_layout(room_type, width, length, height, style=None):
    if room_type not in ["Kitchen Room", "Kitchen", "Bathroom"]:
        raise ValueError("Kitchen/Bathroom generator only supports Kitchen Room, Kitchen, or Bathroom.")

    if kb_df.empty:
        raise ValueError("kitchen_bathroom_metadata_v4_clean.csv is missing or empty.")

    recent_used_ids = get_recently_used_model_ids(limit_predictions=20)
    print(f"[Kitchen Room/Bathroom] recent used count = {len(recent_used_ids)}")

    items = select_kitchen_bathroom_items(room_type, recent_used_ids, style=style)

    if not items:
        raise ValueError(f"No Kitchen Room/Bathroom metadata items available for room type: {room_type}")

    # Large objects first
    def placement_priority(row):
        category = safe_str(row.get("category"))
        fp = get_object_footprint(category)
        return -(fp["width"] * fp["depth"])

    items = sorted(items, key=placement_priority)

    predictions = []

    for row in items:
        category_clean = get_kb_category(row)
        category = safe_str(row.get("category"), category_clean)
        role = safe_str(row.get("anchor_type"), "secondary")
        zone = safe_str(row.get("placement_zone"), "general")

        x_norm, z_norm, y_norm, rotation_y, rule_zone = kb_rule_position(
            room_type=room_type,
            category_clean=category_clean,
            zone=zone
        )

        x_norm = clamp01(x_norm + random.uniform(-0.025, 0.025))
        z_norm = clamp01(z_norm + random.uniform(-0.025, 0.025))

        layout_scale = clamp(safe_float(row.get("layout_scale"), 1.0), 0.5, 1.8)

        pred = {
            "model_id": str(row["model_id"]).strip(),
            "category": category,
            "category_clean": category_clean,
            "style": safe_str(row.get("style"), "Others"),
            "super_category": map_kb_super_category(category_clean, role),
            "x_norm": x_norm,
            "y_norm": clamp01(y_norm),
            "z_norm": z_norm,
            "rotation_x": 0.0,
            "rotation_y": rotation_y,
            "rotation_z": 0.0,
            "scale_x": round(layout_scale, 3),
            "scale_y": round(layout_scale, 3),
            "scale_z": round(layout_scale, 3),
            "role": role,
            "zone": rule_zone,
            "generation_method": "rule_based_kitchen_bathroom",
            "recommendations": extract_recommendations(row),
        }

        pred = kb_collision_adjust(pred, predictions)

        # FIX: compute x/z metres AFTER collision adjustment so they
        # always match x_norm/z_norm. Previously x/z were computed before
        # collision, leaving them inconsistent with the adjusted norms.
        pred["x"] = round(room_scale_x(pred["x_norm"], width), 3)
        pred["y"] = round(pred["y_norm"] * height, 3)
        pred["z"] = round(room_scale_z(pred["z_norm"], length), 3)

        # FIX: smart metadata applies to x_norm/z_norm, then x/z are
        # recomputed for consistency. apply_smart_metadata now updates
        # x_norm/z_norm and we recompute x/z here after the call.
        pred = apply_smart_metadata(pred)
        pred["x"] = round(room_scale_x(pred["x_norm"], width), 3)
        pred["y"] = round(pred["y_norm"] * height, 3)
        pred["z"] = round(room_scale_z(pred["z_norm"], length), 3)

        predictions.append(pred)

        print(
            f"[KB Selected] model_id={pred['model_id']} | "
            f"category={pred['category']} | clean={pred['category_clean']} | "
            f"zone={pred['zone']} | rot={pred['rotation_y']:.0f} | "
            f"x_norm={pred['x_norm']:.3f} | z_norm={pred['z_norm']:.3f} | "
            f"y_norm={pred['y_norm']:.3f} | scale={layout_scale:.3f}"
        )

    return predictions


# =========================================================
# RECOMMENDER OUTPUT HELPERS
# =========================================================
def parse_price_range_to_estimate(price_text, default_price=0):
    if price_text is None:
        return float(default_price)

    text = str(price_text)
    text = text.replace("₱", "").replace("PHP", "").replace("php", "").replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if not numbers:
        return float(default_price)

    values = [float(x) for x in numbers]
    return values[0] if len(values) == 1 else min(values)


def extract_recommendations(row):
    recommendations = []

    for rank in [1, 2, 3]:
        shop = safe_str(row.get(f"recommendation_{rank}_shop"), "")
        link = safe_str(row.get(f"recommendation_{rank}_link"), "")
        availability = safe_str(row.get(f"recommendation_{rank}_availability"), "")
        place = safe_str(row.get(f"recommendation_{rank}_physical_store_place"), "")
        price = safe_str(row.get(f"recommendation_{rank}_price_ph_range"), "")
        notes = safe_str(row.get(f"recommendation_{rank}_notes"), "")
        source = safe_str(row.get(f"recommendation_{rank}_source"), "")

        if not shop or shop == "unknown":
            continue

        estimated_price = parse_price_range_to_estimate(price, default_price=0)

        recommendations.append({
            "rank": rank,
            "shop": shop,
            "price_ph_range": price,
            "estimated_price_php": round(estimated_price, 2),
            "link": link,
            "availability": availability,
            "physical_store_place": place,
            "notes": notes,
            "source": source,
        })

    return recommendations


def get_furniture_importance(pred):
    role = safe_str(pred.get("role"), "secondary").lower()
    super_category = safe_str(pred.get("super_category"), "").lower()
    category = safe_str(pred.get("category"), "").lower()

    if role in ["bed_anchor", "sofa_anchor", "table_anchor"]:
        return 1
    if super_category in ["appliance", "plumbing"]:
        return 1
    if role in ["storage", "fixture", "appliance"]:
        return 2
    if role == "seat" or "chair" in category:
        return 3
    if "lighting" in super_category or role == "light":
        return 4
    return 5


def choose_recommendation_for_item(recommendations, remaining_budget):
    if not recommendations:
        return None

    valid = [r for r in recommendations if safe_float(r.get("estimated_price_php"), 0) > 0]

    if not valid:
        return recommendations[0]

    valid = sorted(valid, key=lambda r: safe_float(r.get("estimated_price_php"), 999999999))
    within_budget = [r for r in valid if safe_float(r.get("estimated_price_php"), 0) <= remaining_budget]

    return within_budget[0] if within_budget else valid[0]


def apply_allocated_budget_to_predictions(predictions, allocated_budget_php=None):
    allocated_budget_php = safe_float(allocated_budget_php, 0)

    for pred in predictions:
        pred["budget_priority"] = get_furniture_importance(pred)
        pred["selected_recommendation"] = None
        pred["selected_price_php"] = 0.0
        pred["included_in_budget"] = False
        pred["budget_note"] = "not_processed"

    if allocated_budget_php <= 0:
        total = 0.0
        for pred in predictions:
            selected = choose_recommendation_for_item(
                pred.get("recommendations", []),
                remaining_budget=999999999
            )
            pred["selected_recommendation"] = selected
            pred["selected_price_php"] = safe_float(selected.get("estimated_price_php"), 0) if selected else 0.0
            pred["included_in_budget"] = True if selected else False
            pred["budget_note"] = "no_allocated_budget_provided"
            total += pred["selected_price_php"]

        return {
            "allocated_budget_php": None,
            "estimated_total_php": round(total, 2),
            "remaining_budget_php": None,
            "budget_status": "no_allocated_budget_provided",
            "is_within_budget": None,
            "included_items": len([p for p in predictions if p.get("included_in_budget")]),
            "excluded_items": 0,
            "excluded_item_categories": [],
        }

    remaining = allocated_budget_php
    total_selected = 0.0

    ordered = sorted(
        predictions,
        key=lambda p: (p.get("budget_priority", 99), safe_str(p.get("category"), ""))
    )

    for pred in ordered:
        selected = choose_recommendation_for_item(
            pred.get("recommendations", []),
            remaining_budget=remaining
        )

        if not selected:
            pred["budget_note"] = "no_recommender_available"
            continue

        price = safe_float(selected.get("estimated_price_php"), 0)

        if price <= remaining:
            pred["selected_recommendation"] = selected
            pred["selected_price_php"] = price
            pred["included_in_budget"] = True
            pred["budget_note"] = "included_within_allocated_budget"
            remaining -= price
            total_selected += price
        else:
            pred["selected_recommendation"] = selected
            pred["selected_price_php"] = price
            pred["included_in_budget"] = False
            pred["budget_note"] = "excluded_over_allocated_budget"

    included_items = [p for p in predictions if p.get("included_in_budget")]
    excluded_items = [p for p in predictions if not p.get("included_in_budget")]

    return {
        "allocated_budget_php": round(allocated_budget_php, 2),
        "estimated_total_php": round(total_selected, 2),
        "remaining_budget_php": round(remaining, 2),
        "budget_status": "within_allocated_budget",
        "is_within_budget": True,
        "included_items": len(included_items),
        "excluded_items": len(excluded_items),
        "excluded_item_categories": [safe_str(p.get("category"), "unknown") for p in excluded_items],
    }


# =========================================================
# MAIN PREDICTION
# =========================================================
def predict_full_room_preview(
    user_room_type=None, style=None,
    width=4.0, length=5.0, height=2.8,
    image_url="", allocated_budget_php=None
):
    room_type = user_room_type if user_room_type else "Bedroom"
    room_type = str(room_type).strip()

    room_type_aliases = {
        "Kitchen": "Kitchen Room",
        "KitchenRoom": "Kitchen Room",
        "Kitchen room": "Kitchen Room",
        "Kitchen Room": "Kitchen Room",
        "Bath": "Bathroom",
        "Bath Room": "Bathroom",
        "Bathroom Room": "Bathroom",
        "Toilet": "Bathroom",
    }
    room_type = room_type_aliases.get(room_type, room_type)

    width, length, height = enforce_code_minimums(width, length, height, room_type)
    scene_meta = analyze_room_image_metadata(image_url)

    # FIX: image-aware correction indentation was broken in original —
    # the soft-correction assignments were outside the if-block, meaning
    # they ran unconditionally and corrupted width/length even when
    # room_depth_confidence was low. Now correctly gated.
    if scene_meta.get("room_depth_confidence", 0) > 0.7:
        estimated_depth = safe_float(scene_meta.get("estimated_room_depth"), length)
        estimated_width = safe_float(scene_meta.get("estimated_room_width"), width)

        # Soft correction: 70% user value, 30% image estimate
        width = width * 0.7 + estimated_width * 0.3
        length = length * 0.7 + estimated_depth * 0.3

        print(f"[IMAGE] corrected room size → {width:.2f} x {length:.2f}")

    # Kitchen/Bathroom use rule-based generator
    if room_type in ["Kitchen Room", "Kitchen", "Bathroom"]:
        print(f"⚡ Using rule-based generator for {room_type}")

        predictions = generate_kitchen_bathroom_layout(
            room_type=room_type,
            width=width,
            length=length,
            height=height,
            style=style
        )

        budget_summary = apply_allocated_budget_to_predictions(predictions, allocated_budget_php=allocated_budget_php)

        return {
            "room_type": room_type,
            "room_width": round(width, 3),
            "room_length": round(length, 3),
            "room_height": round(height, 3),
            "room_area_sqm": round(width * length, 2),
            "budget_summary": budget_summary,
            "furniture": predictions,
        }

    df_filtered = df_master.copy()

    if style and "style" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["style"] == style]

    if df_filtered.empty:
        df_filtered = df_master.copy()

    df_filtered = clean_dataset(df_filtered)

    if df_filtered.empty:
        raise ValueError("No valid furniture models available after cleaning.")

    recent_used_ids = get_recently_used_model_ids(limit_predictions=20)
    print(f"[History] recent used count = {len(recent_used_ids)}")

    items = select_items_for_room(df_filtered, room_type, recent_used_ids)
    if not items:
        raise ValueError(f"No items available for room type: {room_type}")

    predictions = []
    anchors = get_anchor_registry()
    relation_counts = {}

    for row in items:
        category = safe_str(row.get("category"))
        super_category = safe_str(row.get("super-category"))
        zone = safe_str(row.get("placement_zone"), "general")
        role = safe_str(row.get("anchor_type"), "secondary")

        category_key = safe_str(row.get("category", "Other"))
        relation_index = relation_counts.get(category_key, 0)
        relation_counts[category_key] = relation_index + 1

        ai_z_norm = predict_ai_depth(row, room_type, width, length, height)
        ai_scale = predict_ai_scale(row, room_type, width, length, height)

        x_norm, z_norm, _template_rotation_y = generate_spatial_layout(
            item=row,
            anchors=anchors,
            scene_meta=scene_meta,
            room_type=room_type
        )

        x_norm += random.uniform(-0.05, 0.05)
        z_norm += random.uniform(-0.05, 0.05)
        x_norm = clamp01(x_norm)
        z_norm = clamp01(z_norm)

        layout_scale = apply_layout_scale(role, ai_scale, width, length, height)
        y_norm = compute_y_norm(role, category, height)

        smart_rotation_y = get_advanced_rotation(
            category=category,
            super_category=super_category,
            role=role,
            zone=zone,
            x_norm=x_norm,
            z_norm=z_norm,
            anchors=anchors
        )

        pred = {
            "model_id": str(row["model_id"]).strip(),
            "category": category,
            "super_category": super_category,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "z_norm": z_norm,
            "rotation_x": 0.0,
            "rotation_y": smart_rotation_y,
            "rotation_z": 0.0,
            "scale_x": round(layout_scale, 3),
            "scale_y": round(layout_scale, 3),
            "scale_z": round(layout_scale, 3),
            "role": role,
            "zone": zone,
            "recommendations": extract_recommendations(row),
        }

        pred = normalized_collision_adjust(pred, predictions, room_type)

        # FIX: compute x/z metres AFTER collision so they always equal
        # x_norm * room_width (previously computed before collision).
        pred["x"] = round(room_scale_x(pred["x_norm"], width), 3)
        pred["y"] = round(pred["y_norm"] * height, 3)
        pred["z"] = round(room_scale_z(pred["z_norm"], length), 3)

        # FIX: apply smart metadata to norms, then recompute metres.
        pred = apply_smart_metadata(pred)
        pred["x"] = round(room_scale_x(pred["x_norm"], width), 3)
        pred["y"] = round(pred["y_norm"] * height, 3)
        pred["z"] = round(room_scale_z(pred["z_norm"], length), 3)

        predictions.append(pred)
        store_anchor(anchors, pred)

        print(
            f"[Selected] model_id={pred['model_id']} | "
            f"super={pred['super_category']} | category={pred['category']} | "
            f"zone={pred['zone']} | rot={pred['rotation_y']:.0f} | "
            f"x_norm={pred['x_norm']:.3f} | z_norm={pred['z_norm']:.3f} | "
            f"y_norm={pred['y_norm']:.3f} | scale={layout_scale:.3f}"
        )

    budget_summary = apply_allocated_budget_to_predictions(predictions, allocated_budget_php=allocated_budget_php)

    return {
        "room_type": room_type,
        "room_width": round(width, 3),
        "room_length": round(length, 3),
        "room_height": round(height, 3),
        "room_area_sqm": round(width * length, 2),
        "budget_summary": budget_summary,
        "furniture": predictions,
    }


# =========================================================
# REQUEST LOOP
# =========================================================
def process_firestore_requests():
    load_datasets()
    get_models()
    docs = request_collection.where("status", "==", "pending").get()
    if not docs:
        print("No pending requests.")
        return

    for doc in docs:
        try:
            doc_id = doc.id
            data = doc.to_dict()

            print(f"\nProcessing: {doc_id}")
            request_collection.document(doc_id).update({"status": "processing"})

            room_type = data.get("room_type") or data.get("roomName") or "Bedroom"
            style = data.get("style")
            width = float(data.get("room_width", data.get("room_width_m", 4.0)))
            length = float(data.get("room_length", data.get("room_length_m", 5.0)))
            height = float(data.get("room_height", data.get("room_height_m", 2.8)))
            image_url = data.get("imageUrl", "")

            allocated_budget_php = (
                data.get("allocated_budget")
                or data.get("allocatedBudget")
                or data.get("allocated_budget_php")
                or data.get("budget_php")
                or data.get("budget")
                or data.get("max_budget")
                or data.get("room_budget")
                or data.get("total_budget_php")
            )

            prediction = predict_full_room_preview(
                user_room_type=room_type,
                style=style,
                width=width,
                length=length,
                height=height,
                image_url=image_url,
                allocated_budget_php=allocated_budget_php
            )
            prediction["imageUrl"] = image_url

            prediction_collection.document(doc_id).set({
                "status": "completed",
                "prediction_results": prediction,
                "timestamp_completed": firestore.SERVER_TIMESTAMP,
                "timestamp_completed_unix": datetime.now(timezone.utc).timestamp()
            })

            request_collection.document(doc_id).update({"status": "completed"})
            print(f"✅ Saved prediction for {doc_id}")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            traceback.print_exc()

            request_collection.document(doc.id).update({
                "status": "error",
                "error_message": str(e),
            })


from threading import Thread


def firestore_worker():
    print("✅ Phase 3 v6 semantic-lighting recommender app started.")

    while True:
        try:
            process_firestore_requests()
        except Exception as e:
            print(f"Worker error: {e}")

        time.sleep(POLL_SECONDS)


@app.route("/")
def home():
    return "RoomVisualizer AI Worker Running"


if __name__ == "__main__":
    try:
        print("🚀 Starting Flask app...")

        worker_thread = Thread(target=firestore_worker)
        worker_thread.daemon = True
        worker_thread.start()

        port = int(os.environ.get("PORT", 10000))

        app.run(
            host="0.0.0.0",
            port=port,
            debug=True
        )

    except Exception as e:
        import traceback
        print("❌ APP START FAILED:")
        traceback.print_exc()
