import math
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("turn_algo")

# 각도/거리 기준(튜닝 가능 – 기존 값 유지 느낌으로)
ANGLE_UTURN = 150.0          # U턴으로 볼 최소 각도
ANGLE_TURN = 35.0            # 좌/우회전으로 볼 최소 각도
MIN_DIST_TURN = 60.0         # 직전 턴 이후 최소 거리(m)
MIN_DIST_SIMPLIFY = 15.0     # polyline 단순화 최소 거리(m)
MIN_STRAIGHT_SEG = 40.0      # 직진 안내를 줄 최소 구간 길이(m)

RUNNING_SPEED_KMH = 8.0      # 요약용 러닝 속도


# -----------------------------
# 기초 유틸
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
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        total += haversine_m(lat1, lon1, lat2, lon2)
    return total


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """p1 → p2 방위각 (deg, 0=북)."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """두 방위각 차이의 절대값(0~180)."""
    d = abs(a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def simplify_polyline(points: List[Tuple[float, float]], min_dist: float) -> List[Tuple[float, float]]:
    """단순 거리 기반 polyline 축소 (연속점 사이 min_dist 미만이면 생략)."""
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
# 턴바이턴 생성
# -----------------------------

def _classify_turn(prev_b: float, cur_b: float) -> str:
    """방향 변화량을 바탕으로 좌/우/U턴 분류."""
    diff = (cur_b - prev_b) % 360.0
    if diff > 180.0:
        diff -= 360.0  # -180~180

    adiff = abs(diff)

    if adiff >= ANGLE_UTURN:
        return "uturn"
    if adiff < ANGLE_TURN:
        return "straight"
    # 양수면 우회전(오른쪽), 음수면 좌회전(왼쪽) 기준
    return "right" if diff > 0 else "left"


def _build_instruction(turn_type: str, distance_m: float) -> str:
    d = int(round(distance_m))
    if turn_type == "uturn":
        return f"{d}m 앞에서 U턴"
    if turn_type == "left":
        return f"{d}m 앞에서 좌회전"
    if turn_type == "right":
        return f"{d}m 앞에서 우회전"
    # straight
    return f"{d}m 직진"


def build_turn_by_turn(
    polyline: List[Tuple[float, float]],
    km_requested: float,
) -> Tuple[List[Dict], Dict]:
    """polyline 기준 턴바이턴 안내 및 요약 생성."""
    if len(polyline) < 2:
        return [], {
            "length_m": 0.0,
            "km_requested": km_requested,
            "estimated_time_min": 0.0,
            "turn_count": 0,
        }

    # 1) 전체 길이 계산
    total_length_m = polyline_length_m(polyline)

    # 2) polyline 단순화
    simp = simplify_polyline(polyline, MIN_DIST_SIMPLIFY)
    if len(simp) < 2:
        simp = polyline[:]

    turns: List[Dict] = []

    # 3) 전체 경로를 따라가며 누적 거리와 턴 계산
    cum_dist = 0.0
    last_turn_at = 0.0  # 마지막 턴 발생 지점까지 거리
    last_pt = simp[0]
    prev_bearing = None

    for i in range(1, len(simp)):
        cur_pt = simp[i]
        seg_d = haversine_m(*last_pt, *cur_pt)
        cum_dist += seg_d

        if prev_bearing is not None:
            new_bearing = bearing_deg(*last_pt, *cur_pt)
            diff = angle_diff_deg(prev_bearing, new_bearing)

            if diff >= ANGLE_TURN and (cum_dist - last_turn_at) >= MIN_DIST_TURN:
                turn_type = _classify_turn(prev_bearing, new_bearing)
                if turn_type != "straight":
                    turns.append(
                        {
                            "type": turn_type,
                            "lat": cur_pt[0],
                            "lng": cur_pt[1],
                            "at_dist_m": round(cum_dist, 1),
                            "instruction": _build_instruction(
                                turn_type, cum_dist - last_turn_at
                            ),
                        }
                    )
                    last_turn_at = cum_dist

            prev_bearing = new_bearing
        else:
            prev_bearing = bearing_deg(*last_pt, *cur_pt)

        last_pt = cur_pt

    # 4) 너무 직선인 경우, 중간에 한 번 정도 '직진' 안내 추가
    if not turns and total_length_m >= MIN_STRAIGHT_SEG:
        mid_idx = len(simp) // 2
        mid_pt = simp[mid_idx]
        turns.append(
            {
                "type": "straight",
                "lat": mid_pt[0],
                "lng": mid_pt[1],
                "at_dist_m": round(total_length_m / 2.0, 1),
                "instruction": f"{int(round(total_length_m))}m 코스를 따라 직진",
            }
        )

    estimated_time_min = (total_length_m / 1000.0) / (RUNNING_SPEED_KMH / 60.0)
    turn_count = len(
        [t for t in turns if t["type"] in ("straight", "left", "right", "uturn")]
    )

    summary = {
        "length_m": round(total_length_m, 1),
        "km_requested": km_requested,
        "estimated_time_min": round(estimated_time_min, 1),
        "turn_count": turn_count,
    }

    return turns, summary
