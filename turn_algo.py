import math
import logging
from typing import List, Dict, Tuple, Optional

from poi.poi_db import get_nearest_poi  # 로컬 POI DB 조회

logger = logging.getLogger("turn_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 설정값
# -----------------------------
ANGLE_UTURN = 150.0        # U턴 기준 각도
ANGLE_TURN = 20.0          # 턴 감지 기준 각도
MIN_DIST_TURN = 40.0       # 연속된 턴 사이 최소 거리(m)
MIN_DIST_SIMPLIFY = 8.0    # polyline 단순화 간격(m)

CHECKPOINT_INTERVAL = 200.0  # 체크포인트 간격(m)
PRE_ALERT_DIST = 150.0       # 턴 150m 전 예고
EXEC_ALERT_DIST = 30.0       # 턴 30m 전 실행
AFTER_TURN_DIST = 10.0       # 턴 후 10m 피드백

RUNNING_SPEED_KMH = 8.0


# -----------------------------
# 기본 유틸 함수
# -----------------------------
def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """두 위경도 좌표 간 거리(m)."""
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
    """단순 거리 기반 polyline 축소."""
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
# 턴 분류 및 한국어 방향표현
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
    return "right" if diff > 0 else "left"


def _turn_korean(turn_type: str) -> str:
    if turn_type == "left":
        return "좌회전"
    if turn_type == "right":
        return "우회전"
    if turn_type == "uturn":
        return "U턴"
    return "직진"


# -----------------------------
# polyline 상의 특정 거리 위치 보간
# -----------------------------
def interpolate_point_along_polyline(
    points: List[Tuple[float, float]],
    target_dist: float
) -> Tuple[float, float]:
    """
    전체 polyline을 따라 target_dist 위치의 (lat, lng)를 근사적으로 계산.
    polyline 길이보다 크면 마지막 점 반환.
    """
    if not points:
        return (0.0, 0.0)
    if len(points) == 1 or target_dist <= 0:
        return points[0]

    cum = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        seg = haversine_m(lat1, lon1, lat2, lon2)
        if cum + seg >= target_dist:
            ratio = (target_dist - cum) / seg if seg > 0 else 0.0
            lat = lat1 + (lat2 - lat1) * ratio
            lon = lon1 + (lon2 - lon1) * ratio
            return (lat, lon)
        cum += seg

    return points[-1]


# -----------------------------
# 턴바이턴 + 내비게이션 이벤트 생성
# -----------------------------
def build_turn_by_turn(
    polyline: List[Tuple[float, float]],
    km_requested: float,
) -> Tuple[List[Dict], Dict]:
    """
    polyline 기준 턴바이턴 + 음성 이벤트 + 요약 생성.
    반환:
      events: 내비게이션 이벤트(voice events)
      summary: 길이, 예상시간, 이벤트 개수 등
    """
    if len(polyline) < 2:
        summary = {
            "length_m": 0.0,
            "km_requested": km_requested,
            "estimated_time_min": 0.0,
            "event_count": 0,
        }
        return [], summary

    # 전체 길이
    total_length_m = polyline_length_m(polyline)

    # polyline 단순화
    simp = simplify_polyline(polyline, MIN_DIST_SIMPLIFY)
    if len(simp) < 2:
        simp = polyline[:]

    # -------------------------
    # 1) 턴 후보 탐지 (raw turns)
    # -------------------------
    raw_turns = []  # {index, lat, lng, dist_m, type}
    cum_dist = 0.0
    last_turn_at = 0.0
    prev_bearing: Optional[float] = None

    last_pt = simp[0]

    # 체크포인트를 위한 거리 추적
    checkpoints_raw = []  # {dist_m}
    next_checkpoint = CHECKPOINT_INTERVAL

    for i in range(1, len(simp)):
        cur_pt = simp[i]
        seg_d = haversine_m(*last_pt, *cur_pt)
        cum_dist += seg_d

        # 1-1) 체크포인트(200m 간격) 기록(거리만)
        while cum_dist >= next_checkpoint:
            checkpoints_raw.append({"dist_m": next_checkpoint})
            next_checkpoint += CHECKPOINT_INTERVAL

        # 1-2) 턴 감지
        if prev_bearing is not None:
            new_bearing = bearing_deg(*last_pt, *cur_pt)
            diff = angle_diff_deg(prev_bearing, new_bearing)

            if diff >= ANGLE_TURN and (cum_dist - last_turn_at) >= MIN_DIST_TURN:
                turn_type = _classify_turn(prev_bearing, new_bearing)
                if turn_type != "straight":
                    raw_turns.append(
                        {
                            "index": i,
                            "lat": cur_pt[0],
                            "lng": cur_pt[1],
                            "dist_m": cum_dist,
                            "type": turn_type,
                        }
                    )
                    last_turn_at = cum_dist

            prev_bearing = new_bearing
        else:
            prev_bearing = bearing_deg(*last_pt, *cur_pt)

        last_pt = cur_pt

    # -------------------------
    # 2) 각 턴에 대해 POI 조회 (가장 가까운 1개)
    # -------------------------
    poi_cache: Dict[Tuple[int, int], Optional[str]] = {}
    for t in raw_turns:
        lat, lng = t["lat"], t["lng"]
        key = (round(lat * 10000), round(lng * 10000))
        if key in poi_cache:
            poi_name = poi_cache[key]
        else:
            poi_info = get_nearest_poi(lat, lng, radius_m=40.0)
            poi_name = poi_info[0] if poi_info else None
            poi_cache[key] = poi_name
        t["poi"] = poi_name

    # -------------------------
    # 3) 이벤트(voice events) 생성
    # -------------------------
    events: List[Dict] = []

    # (0) 출발 안내
    events.append(
        {
            "type": "start",
            "lat": simp[0][0],
            "lng": simp[0][1],
            "at_dist_m": 0.0,
            "instruction": "러닝을 시작합니다. 계속 직진하세요.",
        }
    )

    raw_turns.sort(key=lambda x: x["dist_m"])

    # (1) 체크포인트 이벤트 생성 (200m, 400m, ...)
    for cp in checkpoints_raw:
        d = cp["dist_m"]
        lat, lng = interpolate_point_along_polyline(simp, d)

        # 이 체크포인트 이후 첫 번째 턴을 찾아서 '다음 방향' 안내
        next_turn = None
        for t in raw_turns:
            if t["dist_m"] >= d:
                next_turn = t
                break

        if next_turn:
            dir_kor = _turn_korean(next_turn["type"])
            msg = f"지금까지 {int(d)}m 이동했습니다. 다음은 {dir_kor}입니다. 계속 직진하세요."
        else:
            msg = f"지금까지 {int(d)}m 이동했습니다. 코스를 따라 계속 직진하세요."

        events.append(
                {
                "type": "checkpoint",
                "lat": lat,
                "lng": lng,
                "at_dist_m": round(d, 1),
                "instruction": msg,
            }
        )

    # (2) 각 턴마다 pre-alert, execution, after-turn 이벤트 생성
    for t in raw_turns:
        dist_turn = t["dist_m"]
        lat = t["lat"]
        lng = t["lng"]
        turn_type = t["type"]
        poi = t.get("poi")
        dir_kor = _turn_korean(turn_type)

        # 2-1) Pre-alert (150m 전)
        pre_dist = dist_turn - PRE_ALERT_DIST
        if pre_dist > 0:
            pre_lat, pre_lng = interpolate_point_along_polyline(simp, pre_dist)
            if poi:
                msg = f"{int(PRE_ALERT_DIST)}m 앞에서 {poi} 지나고 {dir_kor}입니다. 계속 직진하세요."
            else:
                msg = f"{int(PRE_ALERT_DIST)}m 앞에서 {dir_kor}입니다. 계속 직진하세요."

            events.append(
                {
                    "type": f"pre_{turn_type}",
                    "lat": pre_lat,
                    "lng": pre_lng,
                    "at_dist_m": round(pre_dist, 1),
                    "instruction": msg,
                }
            )

        # 2-2) Execution (30m 전 → '이제 ~하세요')
        exec_dist = dist_turn - EXEC_ALERT_DIST
        if exec_dist < 0:
            exec_dist = dist_turn
        exec_lat, exec_lng = interpolate_point_along_polyline(simp, exec_dist)

        if poi:
            msg = f"이제 {poi} 지나서 {dir_kor}하세요."
        else:
            msg = f"이제 {dir_kor}하세요."

        events.append(
            {
                "type": turn_type,
                "lat": exec_lat,
                "lng": exec_lng,
                "at_dist_m": round(exec_dist, 1),
                "instruction": msg,
            }
        )

        # 2-3) After-turn (10m 후)
        after_dist = dist_turn + AFTER_TURN_DIST
        if after_dist <= total_length_m:
            after_lat, after_lng = interpolate_point_along_polyline(simp, after_dist)
            msg = "잘하셨어요. 다음 안내까지 직진하세요."

            events.append(
                {
                    "type": "after_turn",
                    "lat": after_lat,
                    "lng": after_lng,
                    "at_dist_m": round(after_dist, 1),
                    "instruction": msg,
                }
            )

    # -------------------------
    # 4) 이벤트 정렬
    # -------------------------
    events.sort(key=lambda e: e.get("at_dist_m", 0.0))

    # -------------------------
    # summary 생성
    # -------------------------
    estimated_time_min = (total_length_m / 1000.0) / (RUNNING_SPEED_KMH / 60.0)

    summary = {
        "length_m": round(total_length_m, 1),
        "km_requested": km_requested,
        "estimated_time_min": round(estimated_time_min, 1),
        "event_count": len(events),
    }

    return events, summary
