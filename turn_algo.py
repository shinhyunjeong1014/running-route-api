import math
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("turn_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 설정값 (요구사항 적용)
# -----------------------------
ANGLE_UTURN = 150.0      # U턴 기준
ANGLE_TURN = 20.0        # ← 요구사항: 턴 감지 기준 20도
MIN_DIST_TURN = 40.0     # 연속된 턴 최소 간격(기존 60 → 40으로 살짝 완화)
MIN_DIST_SIMPLIFY = 8.0  # polyline 단순화(기존 15 → 8)
CHECKPOINT_INTERVAL = 200.0  # ← 요구사항: 200m 마다 안내

RUNNING_SPEED_KMH = 8.0


# -----------------------------
# 기본 유틸
# -----------------------------

def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lng2 - lng1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def polyline_length_m(points: List[Tuple[float, float]]) -> float:
    return sum(
        haversine_m(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
        for i in range(len(points)-1)
    ) if len(points) >= 2 else 0.0


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)

    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def simplify_polyline(points: List[Tuple[float, float]], min_dist: float):
    if len(points) < 2:
        return points[:]

    simplified = [points[0]]
    acc_dist = 0.0

    for i in range(1, len(points)):
        d = haversine_m(*simplified[-1], *points[i])
        acc_dist += d
        if acc_dist >= min_dist:
            simplified.append(points[i])
            acc_dist = 0.0

    if simplified[-1] != points[-1]:
        simplified.append(points[-1])

    return simplified


# -----------------------------
# 턴 분류
# -----------------------------

def _classify_turn(prev_b: float, cur_b: float) -> str:
    diff = (cur_b - prev_b) % 360.0
    if diff > 180.0:
        diff -= 360.0

    adiff = abs(diff)

    if adiff >= ANGLE_UTURN:
        return "uturn"
    if adiff < ANGLE_TURN:
        return "straight"

    return "right" if diff > 0 else "left"


def _turn_instruction(turn_type: str, dist_m: float) -> str:
    d = int(round(dist_m))
    if turn_type == "uturn":
        return f"약 {d}m 후 U턴하세요."
    if turn_type == "left":
        return f"약 {d}m 후 좌회전하세요."
    if turn_type == "right":
        return f"약 {d}m 후 우회전하세요."
    return f"약 {d}m 직진하세요."


# -----------------------------
# 턴바이턴 생성 + 200m 체크포인트 기능 추가
# -----------------------------

def build_turn_by_turn(polyline: List[Tuple[float, float]], km_requested: float):
    if len(polyline) < 2:
        return [], {
            "length_m": 0.0,
            "estimated_time_min": 0.0,
            "turn_count": 0,
            "km_requested": km_requested,
        }

    # 전체 거리
    total_length_m = polyline_length_m(polyline)

    # polyline 단순화
    simp = simplify_polyline(polyline, MIN_DIST_SIMPLIFY)

    turns = []
    cum_dist = 0.0
    last_turn_at = 0.0
    next_checkpoint = CHECKPOINT_INTERVAL

    prev_bearing: Optional[float] = None
    last_pt = simp[0]

    for i in range(1, len(simp)):
        cur_pt = simp[i]
        seg_d = haversine_m(*last_pt, *cur_pt)
        cum_dist += seg_d

        # ---------------------------
        # ① 체크포인트(200m) 자동 안내
        # ---------------------------
        while cum_dist >= next_checkpoint:
            turns.append({
                "type": "checkpoint",
                "lat": cur_pt[0],
                "lng": cur_pt[1],
                "at_dist_m": next_checkpoint,
                "instruction": f"{int(next_checkpoint)}m 직진하세요."
            })
            next_checkpoint += CHECKPOINT_INTERVAL

        # ---------------------------
        # ② 턴 감지
        # ---------------------------
        if prev_bearing is not None:
            new_bearing = bearing_deg(*last_pt, *cur_pt)
            diff = angle_diff_deg(prev_bearing, new_bearing)

            if diff >= ANGLE_TURN and (cum_dist - last_turn_at) >= MIN_DIST_TURN:
                turn_type = _classify_turn(prev_bearing, new_bearing)
                if turn_type != "straight":
                    turns.append({
                        "type": turn_type,
                        "lat": cur_pt[0],
                        "lng": cur_pt[1],
                        "at_dist_m": round(cum_dist, 1),
                        "instruction": _turn_instruction(turn_type, cum_dist - last_turn_at)
                    })
                    last_turn_at = cum_dist

            prev_bearing = new_bearing
        else:
            prev_bearing = bearing_deg(*last_pt, *cur_pt)

        last_pt = cur_pt

    # ---------------------------
    # summary
    # ---------------------------
    est_time_min = (total_length_m / 1000.0) / (RUNNING_SPEED_KMH / 60.0)

    summary = {
        "length_m": round(total_length_m, 1),
        "estimated_time_min": round(est_time_min, 1),
        "turn_count": len(turns),
        "km_requested": km_requested,
    }

    return turns, summary
