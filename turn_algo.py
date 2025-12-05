import math
import logging
from typing import List, Dict, Tuple, Optional

import requests

logger = logging.getLogger("turn_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 설정값
# -----------------------------
ANGLE_UTURN = 150.0        # U턴 기준 각도
ANGLE_TURN = 45.0          # 턴 감지 기준 각도
MIN_DIST_TURN = 40.0       # 연속된 턴 사이 최소 거리(m)
MIN_DIST_SIMPLIFY = 8.0    # polyline 단순화 간격(m)

CHECKPOINT_INTERVAL = 200.0        # 체크포인트 간격(m)
CHECKPOINT_TURN_BUFFER = 200.0     # checkpoint ~ 다음 턴 거리 < 200m면 checkpoint 제거

PRE_ALERT_DIST = 150.0     # 턴 150m 전 예고
EXEC_ALERT_DIST = 30.0     # 턴 30m 전 실행
AFTER_TURN_DIST = 10.0     # 턴 후 10m 피드백

RUNNING_SPEED_KMH = 8.0

# POI 관련 (카카오만 사용)
KAKAO_REST_API_KEY = "dc3686309f8af498d7c62bed0321ee64"
KAKAO_POI_RADIUS_M = 150.0        # 턴 주변 검색 반경 (직선 거리)

POI_SCAN_BEFORE_M = 60.0          # 턴 기준 경로 상 이전 구간(POI 허용 범위)
POI_SCAN_AFTER_M = 80.0           # 턴 기준 경로 상 이후 구간(POI 허용 범위)
POI_LATERAL_MAX_M = 25.0          # 경로와 수직거리 임계값(이상일 경우 "지나간다"라고 말하면 안 됨)

KAKAO_TIMEOUT_SEC = 2.0
KAKAO_CATEGORY_CODES = ["CS2", "CE7", "FD6"]  # 편의점, 카페, 음식점 위주

# 브랜드 우선순위 (숫자 작을수록 우선순위 높음)
BRAND_KEYWORDS = [
    ("세븐일레븐", 0),
    ("7-ELEVEN", 0),
    ("GS25", 0),
    ("CU", 0),
    ("이마트24", 0),

    ("스타벅스", 1),
    ("빽다방", 1),
    ("파스쿠찌", 1),
    ("이디야", 1),
    ("메가커피", 1),
    ("컴포즈", 1),
    ("투썸", 1),
    ("할리스", 1),
    ("폴바셋", 1),

    ("맘스터치", 2),
    ("맥도날드", 2),
    ("버거킹", 2),
    ("롯데리아", 2),
    ("KFC", 2),
    ("BBQ", 2),
    ("BHC", 2),
    ("교촌", 2),
]

CATEGORY_RANK = {
    "CS2": 0,  # 편의점
    "CE7": 1,  # 카페
    "FD6": 2,  # 음식점
}


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
# polyline 상에서 임의 좌표의 "경로 상 거리" + 경로와의 수직거리
# -----------------------------
def distance_along_polyline_to_point(
    points: List[Tuple[float, float]],
    lat: float,
    lng: float,
) -> Tuple[float, float]:
    """
    polyline에서 가장 가까운 vertex를 기준으로
    - 그 지점까지의 경로 거리
    - 해당 vertex와의 직선 거리(수직 거리 근사)
    를 반환.
    """
    if not points:
        return 0.0, float("inf")

    best_idx = 0
    best_d = float("inf")
    for i, (plat, plng) in enumerate(points):
        d = haversine_m(lat, lng, plat, plng)
        if d < best_d:
            best_d = d
            best_idx = i

    cum = 0.0
    for i in range(best_idx):
        cum += haversine_m(*points[i], *points[i + 1])

    return cum, best_d


def classify_poi_relation_to_turn(
    poi_dist_along: float,
    turn_dist: float,
    at_threshold: float = 20.0,
) -> str:
    """
    POI가 턴 기준으로
      - 이전(before)
      - 교차로(at)
      - 이후(after)
    인지 판별.
    """
    if poi_dist_along < turn_dist - at_threshold:
        return "before"
    if poi_dist_along > turn_dist + at_threshold:
        return "after"
    return "at"


# -----------------------------
# POI 이름 축약 + 브랜드 우선순위
# -----------------------------
def shorten_poi_name(name: str) -> str:
    name = name.strip()

    # 괄호 뒤, 하이픈 뒤 등 추가정보 제거
    for sep in ["|", "/", "·", "-", "—", "(", ","]:
        if sep in name:
            name = name.split(sep)[0].strip()

    # "OO점", "OO지점", "OO호점" 등의 꼬리 제거
    for suffix in ["지점", "점", "호점"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()

    # 너무 길면 15자 정도로 자르기
    if len(name) > 15:
        name = name[:15].rstrip()
    return name


def brand_priority(name: str) -> int:
    lname = name.lower()
    best = 3  # 기본값(낮은 우선순위)
    for kw, rank in BRAND_KEYWORDS:
        if kw.lower() in lname:
            best = min(best, rank)
    return best


# -----------------------------
# 카카오맵 POI 후보 검색
# -----------------------------
def search_kakao_poi_candidates(
    lat: float,
    lng: float,
    radius_m: float = KAKAO_POI_RADIUS_M,
) -> List[Dict]:
    """
    카카오 카테고리 검색 기반으로 여러 POI 후보를 가져온다.
    반환: documents 리스트(원본 JSON 일부)
    """
    if not KAKAO_REST_API_KEY:
        return []

    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    candidates: List[Dict] = []

    for cat in KAKAO_CATEGORY_CODES:
        try:
            resp = requests.get(
                "https://dapi.kakao.com/v2/local/search/category.json",
                headers=headers,
                params={
                    "category_group_code": cat,
                    "y": lat,
                    "x": lng,
                    "radius": int(radius_m),
                    "sort": "distance",
                    "size": 15,
                },
                timeout=KAKAO_TIMEOUT_SEC,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            for doc in data.get("documents", []):
                doc["category_group_code"] = cat
                candidates.append(doc)
        except Exception as e:
            logger.warning(f"Kakao POI search error: {e}")
            continue

    return candidates


# -----------------------------
# 턴 기준으로 사용할 최적 POI 선택 (카카오만 사용)
# -----------------------------
def choose_best_poi_for_turn(
    simp_polyline: List[Tuple[float, float]],
    turn_dist: float,
    turn_lat: float,
    turn_lng: float,
) -> Optional[Dict]:
    """
    턴 근처에서 사용할 대표 POI 1개 선택.
    반환: {
        "name": str,
        "lat": float,
        "lng": float,
        "category_group_code": str,
        "relation": "before"|"at"|"after",
        "dist_along": float,
        "lateral": float,
    } 또는 None
    """
    docs = search_kakao_poi_candidates(turn_lat, turn_lng, radius_m=KAKAO_POI_RADIUS_M)
    if not docs:
        return None

    best_poi: Optional[Dict] = None
    best_score: Optional[Tuple[float, float, float, float]] = None

    for doc in docs:
        try:
            name_raw = doc.get("place_name")
            if not name_raw:
                continue
            name = shorten_poi_name(name_raw)

            poi_lat = float(doc["y"])
            poi_lng = float(doc["x"])
            cat_code = doc.get("category_group_code", "")

            dist_along, lateral = distance_along_polyline_to_point(
                simp_polyline, poi_lat, poi_lng
            )

            # 1) 경로와 너무 멀면 제외 (회뜨는 포차 같은 케이스 방지)
            if lateral > POI_LATERAL_MAX_M:
                continue

            # 2) 턴 기준 경로 상 구간 필터
            if dist_along < turn_dist - POI_SCAN_BEFORE_M:
                continue
            if dist_along > turn_dist + POI_SCAN_AFTER_M:
                continue

            relation = classify_poi_relation_to_turn(dist_along, turn_dist)

            # 3) 스코어링
            b_rank = brand_priority(name)
            c_rank = CATEGORY_RANK.get(cat_code, 5)
            along_diff = abs(dist_along - turn_dist)

            score = (b_rank, c_rank, along_diff, lateral)

            if (best_score is None) or (score < best_score):
                best_score = score
                best_poi = {
                    "name": name,
                    "lat": poi_lat,
                    "lng": poi_lng,
                    "category_group_code": cat_code,
                    "relation": relation,
                    "dist_along": dist_along,
                    "lateral": lateral,
                }
        except Exception:
            continue

    return best_poi


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
    # 2) 각 턴에 대해 POI 조회 + 관계 계산
    # -------------------------
    for t in raw_turns:
        t_lat = t["lat"]
        t_lng = t["lng"]
        t_dist = t["dist_m"]

        poi = choose_best_poi_for_turn(simp, t_dist, t_lat, t_lng)
        if poi is None:
            t["poi_name"] = None
        else:
            t["poi_name"] = poi["name"]
            t["poi_relation"] = poi["relation"]
            t["poi_dist_along"] = poi["dist_along"]
            t["poi_lateral"] = poi["lateral"]
            t["poi_category_group_code"] = poi["category_group_code"]

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

        # 이 체크포인트 이후 첫 번째 턴을 찾아서 '다음 방향' 안내
        next_turn = None
        for t in raw_turns:
            if t["dist_m"] >= d:
                next_turn = t
                break

        # 턴이 200m 이내에 있으면 체크포인트 이벤트 생성하지 않음
        if next_turn and (next_turn["dist_m"] - d) < CHECKPOINT_TURN_BUFFER:
            continue

        lat, lng = interpolate_point_along_polyline(simp, d)

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
        turn_lat = t["lat"]
        turn_lng = t["lng"]
        turn_type = t["type"]
        poi_name = t.get("poi_name")
        poi_relation = t.get("poi_relation")
        dir_kor = _turn_korean(turn_type)

        # 2-1) Pre-alert (150m 전)
        pre_dist = dist_turn - PRE_ALERT_DIST
        if pre_dist > 0:
            pre_lat, pre_lng = interpolate_point_along_polyline(simp, pre_dist)

            if poi_name:
                if poi_relation == "before":
                    msg = f"{int(PRE_ALERT_DIST)}m 앞 {poi_name} 지나서 {dir_kor}입니다. 계속 직진하세요."
                elif poi_relation == "at":
                    msg = f"{int(PRE_ALERT_DIST)}m 앞 {poi_name}에서 {dir_kor}입니다. 계속 직진하세요."
                else:  # after
                    msg = f"{int(PRE_ALERT_DIST)}m 앞 {dir_kor}입니다. {dir_kor} 후 {poi_name}이 나옵니다. 계속 직진하세요."
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

        if poi_name:
            if poi_relation == "before":
                msg = f"이제 {poi_name} 지나서 {dir_kor}하세요."
            elif poi_relation == "at":
                msg = f"이제 {poi_name}에서 {dir_kor}하세요."
            else:  # after
                msg = f"이제 {dir_kor}하세요. {dir_kor} 후 {poi_name}이 나옵니다."
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
