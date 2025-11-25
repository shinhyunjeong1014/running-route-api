# turn_algo.py
"""
러닝 경로용 턴-바이-턴(음성 안내) 생성 모듈

- polyline(위도/경도 리스트)을 받아서
  * 출발 안내
  * 구간별 "XXXm 직진하세요"
  * 좌/우회전, U턴
  * 도착 안내
  형태의 심플 내비게이션 스타일 안내를 만든다.

스타일:
  - 불필요한 코멘트 없이, 음성 안내에 적합한 짧은 문장만 사용
  - 예) "290m 직진하세요", "좌회전하세요", "우회전하세요", "U턴 하세요"
"""

import math
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("turn_algo")

# 회전 감지 및 안내 설정
ANGLE_UTURN = 150          # U턴 판정 각도 (deg)
ANGLE_TURN = 30            # 좌/우회전 판정 최소 각도 (deg)
MIN_DIST_TURN = 30.0       # turn 사이 최소 거리(m)
MIN_DIST_SIMPLIFY = 12.0   # polyline 간소화 최소 간격(m)
MIN_STRAIGHT_SEG = 20.0    # 직진 안내를 줄 최소 구간(m)


# -----------------------------
# 기본 지오메트리 유틸
# -----------------------------
def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """위도/경도 두 점 사이의 거리 (m)."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lng2 - lng1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def _cumulative_distances(points: List[Tuple[float, float]]) -> List[float]:
    """각 점까지의 누적 거리 리스트."""
    if not points:
        return []
    dists = [0.0]
    for i in range(1, len(points)):
        d = haversine_m(points[i-1][0], points[i-1][1], points[i][0], points[i][1])
        dists.append(dists[-1] + d)
    return dists


def bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """두 점 사이의 방위각 (degrees, 북=0, 시계방향)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lng2 - lng1)

    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    brng = math.degrees(math.atan2(x, y))
    # 0~360
    return (brng + 360.0) % 360.0


def angle_diff_deg(a1: float, a2: float) -> float:
    """
    두 방위각 차이 (deg). 결과는 -180 ~ 180 범위.
    양수: 좌회전, 음수: 우회전 방향.
    """
    diff = (a2 - a1 + 180.0) % 360.0 - 180.0
    return diff


# -----------------------------
# Polyline 간소화
# -----------------------------
def simplify_polyline(polyline: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    지나치게 조밀한 포인트를 제거하여, turn 검출을 안정화한다.
    - 연속 포인트 사이 거리가 MIN_DIST_SIMPLIFY 미만이면 건너뜀.
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


# -----------------------------
# 거리 포맷
# -----------------------------
def _format_distance_m(distance_m: float) -> str:
    """
    안내 문구용 거리 포맷.
    - 0~999m: 10m 단위 반올림 → '290m'
    - 1km 이상: 소수 1자리 km → '1.2km'
    """
    if distance_m < 1000.0:
        d10 = int(round(distance_m / 10.0) * 10)
        if d10 < 10:
            d10 = 10
        return f"{d10}m"
    else:
        km = distance_m / 1000.0
        return f"{km:.1f}km"


# -----------------------------
# 메인: 턴-바이-턴 생성
# -----------------------------
def build_turn_by_turn(
    polyline: List[Dict[str, float]],
    km_requested: float,
):
    """
    Valhalla에서 받은 polyline(위도/경도 리스트)을 바탕으로 심플 턴-바이-턴 안내를 생성한다.

    반환:
        turns: [
          {
            "lat": ...,
            "lng": ...,
            "type": "start" | "straight" | "left" | "right" | "uturn" | "arrive",
            "angle": float,      # start/arrive는 생략 가능, straight는 0.0
            "distance": float,   # 이 안내 구간의 거리 (m)
            "instruction": str,
          },
          ...
        ]

        summary: {
          "length_m": float,
          "km_requested": float,
          "estimated_time_min": float,
          "turn_count": int,   # straight/left/right/uturn 개수
        }
    """
    if not polyline or len(polyline) < 2:
        # 최소한의 형식만 유지
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

    # 1) polyline 간소화
    simp = simplify_polyline(polyline)
    points = [(p["lat"], p["lng"]) for p in simp]
    cum = _cumulative_distances(points)
    total_length_m = cum[-1]

    logger.info("[TurnAlgo] simplified points=%d, total_length=%.1fm", len(points), total_length_m)

    # 2) 회전 지점 탐지
    turn_candidates = []  # (idx, type, angle_deg)
    last_turn_idx = 0

    for i in range(1, len(points) - 1):
        lat0, lng0 = points[i - 1]
        lat1, lng1 = points[i]
        lat2, lng2 = points[i + 1]

        # 너무 짧은 세그먼트는 스킵
        d01 = haversine_m(lat0, lng0, lat1, lng1)
        d12 = haversine_m(lat1, lng1, lat2, lng2)
        if d01 < MIN_DIST_SIMPLIFY or d12 < MIN_DIST_SIMPLIFY:
            continue

        b1 = bearing_deg(lat0, lng0, lat1, lng1)
        b2 = bearing_deg(lat1, lng1, lat2, lng2)
        ang = angle_diff_deg(b1, b2)
        abs_ang = abs(ang)

        if abs_ang < ANGLE_TURN:
            # 작은 각도 변화는 직선으로 간주
            continue

        # turn 간 최소 거리 조건
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

    # 3) 안내 이벤트 생성
    turns: List[Dict] = []
    start_lat, start_lng = points[0]
    # 시작 안내
    turns.append(
        {
            "lat": start_lat,
            "lng": start_lng,
            "type": "start",
            "distance": 0.0,
            "instruction": "러닝을 시작합니다. 직진하세요.",
        }
    )

    # anchor: 마지막으로 안내한 위치의 인덱스
    anchor_idx = 0

    def add_straight_event(from_idx: int, to_idx: int):
        """from_idx → to_idx 구간에 대한 직진 안내 추가."""
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

    # 각 turn 전/후로 직진 + 회전 안내
    for idx, ttype, ang in turn_candidates:
        # 3-1) anchor → turn 지점까지 직진
        add_straight_event(anchor_idx, idx)

        # 3-2) turn 안내
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

    # 4) 마지막 구간 직진 + 도착 안내
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

    # 5) summary 계산
    RUNNING_SPEED_KMH = 8.0  # 8km/h (7분30초~8분 페이스 정도)
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
