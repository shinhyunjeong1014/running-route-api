import math
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("turn_algo")

# 각도/거리 기준(튜닝 가능)
ANGLE_UTURN = 150           # U턴으로 볼 최소 각도
ANGLE_TURN = 35             # 좌/우회전으로 볼 최소 각도 (기존 30 → 35)
MIN_DIST_TURN = 60.0        # 직전 턴 이후 최소 거리 (기존 30 → 60)
MIN_DIST_SIMPLIFY = 15.0    # polyline 단순화를 위한 최소 거리 (기존 12 → 15)
MIN_STRAIGHT_SEG = 40.0     # 직진 안내를 줄 최소 구간 길이 (기존 20 → 40)


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lng2 - lng1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def _cumulative_distances(points: List[Tuple[float, float]]) -> List[float]:
    """각 점까지의 누적 거리 리스트 (0부터 시작)."""
    if not points:
        return []
    dists = [0.0]
    for i in range(1, len(points)):
        d = haversine_m(points[i - 1][0], points[i - 1][1],
                        points[i][0], points[i][1])
        dists.append(dists[-1] + d)
    return dists


def bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """두 점 사이의 방위각(0~360도)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lng2 - lng1)

    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - \
        math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def angle_diff_deg(a1: float, a2: float) -> float:
    """두 방위각의 signed difference (-180~180)."""
    diff = (a2 - a1 + 180.0) % 360.0 - 180.0
    return diff


def simplify_polyline(polyline: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    너무 촘촘한 점들을 제거해서 턴 감지 시 노이즈를 줄이기.
    """
    if len(polyline) <= 2:
        return polyline[:]

    simplified = [polyline[0]]
    last = polyline[0]
    for pt in polyline[1:]:
        d = haversine_m(last["lat"], last["lng"], pt["lat"], pt["lng"])
        if d >= MIN_DIST_SIMPLIFY:
            simplified.append(pt)
            last = pt
    if simplified[-1] is not polyline[-1]:
        simplified.append(polyline[-1])
    return simplified


def _format_distance_m(distance_m: float) -> str:
    """
    음성 안내용 거리 문자열.
    - 1000m 미만: 10m 단위 반올림 → "250m"
    - 1000m 이상: 소수 1자리 km → "1.4km"
    """
    if distance_m < 1000.0:
        d10 = int(round(distance_m / 10.0) * 10)
        if d10 < 10:
            d10 = 10
        return f"{d10}m"
    else:
        km = distance_m / 1000.0
        return f"{km:.1f}km"


def build_turn_by_turn(polyline: List[Dict[str, float]], km_requested: float):
    """
    폴리라인(지도 경로)을 받아 음성 안내용 턴 리스트와 요약 정보를 생성.
    - '직진 x m' + 별도 '좌/우회전' 이벤트 (옵션 2 스타일)
    """
    # 예외 처리: 경로가 거의 없는 경우
    if not polyline or len(polyline) < 2:
        if not polyline:
            return [], {
                "length_m": 0.0,
                "km_requested": km_requested,
                "estimated_time_min": 0.0,
                "turn_count": 0,
            }

        start = polyline[0]
        turns = [
            {
                "lat": start["lat"],
                "lng": start["lng"],
                "type": "start",
                "distance": 0.0,
                "instruction": "러닝을 시작합니다. 직진하세요.",
            },
            {
                "lat": start["lat"],
                "lng": start["lng"],
                "type": "arrive",
                "distance": 0.0,
                "instruction": "목적지에 도착했습니다. 러닝을 완료했습니다.",
            },
        ]
        return turns, {
            "length_m": 0.0,
            "km_requested": km_requested,
            "estimated_time_min": 0.0,
            "turn_count": 0,
        }

    # 1) 폴리라인 단순화
    simp = simplify_polyline(polyline)
    points = [(p["lat"], p["lng"]) for p in simp]
    cum = _cumulative_distances(points)
    total_length_m = cum[-1]

    logger.info("[TurnAlgo] simplified points=%d, total_length=%.1fm",
                len(points), total_length_m)

    # 2) 턴 후보 탐지
    turn_candidates = []
    last_turn_idx = 0

    for i in range(1, len(points) - 1):
        lat0, lng0 = points[i - 1]
        lat1, lng1 = points[i]
        lat2, lng2 = points[i + 1]

        d01 = haversine_m(lat0, lng0, lat1, lng1)
        d12 = haversine_m(lat1, lng1, lat2, lng2)
        if d01 < MIN_DIST_SIMPLIFY or d12 < MIN_DIST_SIMPLIFY:
            continue

        b1 = bearing_deg(lat0, lng0, lat1, lng1)
        b2 = bearing_deg(lat1, lng1, lat2, lng2)
        ang = angle_diff_deg(b1, b2)
        abs_ang = abs(ang)

        # 각도가 너무 작으면 직진으로 간주
        if abs_ang < ANGLE_TURN:
            continue

        # 직전 턴 이후 충분한 거리가 쌓였는지 확인
        dist_since_last = cum[i] - cum[last_turn_idx]
        if dist_since_last < MIN_DIST_TURN:
            continue

        if abs_ang >= ANGLE_UTURN:
            ttype = "uturn"
        else:
            ttype = "left" if ang > 0 else "right"

        turn_candidates.append((i, ttype, ang))
        last_turn_idx = i

    logger.info("[TurnAlgo] detected turns=%d", len(turn_candidates))

    # 3) 이벤트 리스트 구성
    turns: List[Dict] = []
    start_lat, start_lng = points[0]
    turns.append(
        {
            "lat": start_lat,
            "lng": start_lng,
            "type": "start",
            "distance": 0.0,
            "instruction": "러닝을 시작합니다. 직진하세요.",
        }
    )

    anchor_idx = 0  # '직진 시작' 기준 인덱스

    def add_straight_event(from_idx: int, to_idx: int):
        """anchor → 특정 턴 직전까지 '직진 x m' 이벤트 추가."""
        if to_idx <= from_idx:
            return
        seg_len = cum[to_idx] - cum[from_idx]
        if seg_len < MIN_STRAIGHT_SEG:
            return

        lat, lng = points[to_idx]
        dist_text = _format_distance_m(seg_len)
        turns.append(
            {
                "lat": lat,
                "lng": lng,
                "type": "straight",
                "angle": 0.0,
                "distance": round(seg_len, 1),
                "instruction": f"{dist_text} 직진하세요",
            }
        )

    # 각 턴까지 직진 이벤트 → 턴 이벤트 → 앵커 갱신
    for idx, ttype, ang in turn_candidates:
        add_straight_event(anchor_idx, idx)

        lat, lng = points[idx]
        if ttype == "left":
            instr = "좌회전하세요."
        elif ttype == "right":
            instr = "우회전하세요."
        else:
            instr = "U턴 하세요."

        turns.append(
            {
                "lat": lat,
                "lng": lng,
                "type": ttype,
                "angle": round(ang, 1),
                "distance": 0.0,
                "instruction": instr,
            }
        )
        anchor_idx = idx

    # 마지막 턴 이후 → 도착점까지 직진 이벤트
    last_idx = len(points) - 1
    add_straight_event(anchor_idx, last_idx)

    end_lat, end_lng = points[-1]
    turns.append(
        {
            "lat": end_lat,
            "lng": end_lng,
            "type": "arrive",
            "distance": 0.0,
            "instruction": "목적지에 도착했습니다. 러닝을 완료했습니다.",
        }
    )

    # 4) 요약 정보
    RUNNING_SPEED_KMH = 8.0
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
