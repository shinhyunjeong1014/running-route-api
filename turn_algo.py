import math
import logging
import asyncio
from typing import List, Dict, Tuple, Optional

import httpx  # pip install httpx

logger = logging.getLogger("turn_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 설정값
# -----------------------------
ANGLE_UTURN = 150.0
ANGLE_TURN = 45.0
MIN_DIST_TURN = 40.0
MIN_DIST_SIMPLIFY = 8.0

CHECKPOINT_INTERVAL = 200.0
CHECKPOINT_TURN_BUFFER = 200.0

PRE_ALERT_DIST = 150.0
EXEC_ALERT_DIST = 30.0
AFTER_TURN_DIST = 10.0

RUNNING_SPEED_KMH = 8.0

KAKAO_REST_API_KEY = "dc3686309f8af498d7c62bed0321ee64"
KAKAO_POI_RADIUS_M = 150.0
POI_SCAN_BEFORE_M = 60.0
POI_SCAN_AFTER_M = 80.0
POI_LATERAL_MAX_M = 25.0
KAKAO_TIMEOUT_SEC = 2.0

KAKAO_CATEGORY_CODES = ["CS2", "CE7", "FD6"]

BRAND_KEYWORDS = [
    ("세븐일레븐", 0), ("7-ELEVEN", 0), ("GS25", 0), ("CU", 0), ("이마트24", 0),
    ("스타벅스", 1), ("빽다방", 1), ("파스쿠찌", 1), ("이디야", 1), ("메가커피", 1),
    ("컴포즈", 1), ("투썸", 1), ("할리스", 1), ("폴바셋", 1),
    ("맘스터치", 2), ("맥도날드", 2), ("버거킹", 2), ("롯데리아", 2), ("KFC", 2),
    ("BBQ", 2), ("BHC", 2), ("교촌", 2),
]
CATEGORY_RANK = {"CS2": 0, "CE7": 1, "FD6": 2}


# -----------------------------
# 기본 유틸 (math, string) - 동기 함수 유지
# -----------------------------
def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = p2 - p1; dl = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def polyline_length_m(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2: return 0.0
    return sum(haversine_m(*points[i], *points[i+1]) for i in range(len(points)-1))

def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

def angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d

def simplify_polyline(points: List[Tuple[float, float]], min_dist: float) -> List[Tuple[float, float]]:
    if len(points) < 2: return points[:]
    simplified = [points[0]]
    acc = 0.0
    for i in range(1, len(points)):
        d = haversine_m(*simplified[-1], *points[i])
        acc += d
        if acc >= min_dist:
            simplified.append(points[i])
            acc = 0.0
    if simplified[-1] != points[-1]:
        simplified.append(points[-1])
    return simplified

def _classify_turn(prev_b: float, cur_b: float) -> str:
    diff = (cur_b - prev_b) % 360.0
    if diff > 180.0: diff -= 360.0
    adiff = abs(diff)
    if adiff >= ANGLE_UTURN: return "uturn"
    if adiff < ANGLE_TURN: return "straight"
    return "right" if diff > 0 else "left"

def _turn_korean(turn_type: str) -> str:
    return {"left": "좌회전", "right": "우회전", "uturn": "U턴"}.get(turn_type, "직진")

def interpolate_point_along_polyline(points: List[Tuple[float, float]], target_dist: float) -> Tuple[float, float]:
    if not points: return (0.0, 0.0)
    if len(points) == 1 or target_dist <= 0: return points[0]
    cum = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        seg = haversine_m(lat1, lon1, lat2, lon2)
        if cum + seg >= target_dist:
            r = (target_dist - cum) / seg if seg > 0 else 0.0
            return (lat1 + (lat2-lat1)*r, lon1 + (lon2-lon1)*r)
        cum += seg
    return points[-1]

def distance_along_polyline_to_point(points: List[Tuple[float, float]], lat: float, lng: float) -> Tuple[float, float]:
    if not points: return 0.0, float("inf")
    best_idx, best_d = 0, float("inf")
    for i, (plat, plng) in enumerate(points):
        d = haversine_m(lat, lng, plat, plng)
        if d < best_d:
            best_d = d
            best_idx = i
    cum = sum(haversine_m(*points[j], *points[j+1]) for j in range(best_idx))
    return cum, best_d

def classify_poi_relation_to_turn(poi_dist: float, turn_dist: float) -> str:
    if poi_dist < turn_dist - 20.0: return "before"
    if poi_dist > turn_dist + 20.0: return "after"
    return "at"

def shorten_poi_name(name: str) -> str:
    name = name.strip()
    for sep in ["|", "/", "·", "-", "—", "(", ","]:
        name = name.split(sep)[0].strip()
    for s in ["지점", "점", "호점"]:
        if name.endswith(s):
            name = name[:-len(s)].strip()
    return name[:15].rstrip()

def brand_priority(name: str) -> int:
    lname = name.lower()
    best = 3
    for kw, rank in BRAND_KEYWORDS:
        if kw.lower() in lname: best = min(best, rank)
    return best


# -----------------------------
# [Async] 카카오맵 POI 검색
# -----------------------------
async def search_kakao_poi_candidates_async(
    client: httpx.AsyncClient,
    lat: float, lng: float, radius_m: float
) -> List[Dict]:
    if not KAKAO_REST_API_KEY: return []
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    
    # 카테고리별 병렬 요청 준비
    tasks = []
    for cat in KAKAO_CATEGORY_CODES:
        params = {
            "category_group_code": cat,
            "y": lat, "x": lng, "radius": int(radius_m),
            "sort": "distance", "size": 15
        }
        tasks.append(
            client.get(
                "https://dapi.kakao.com/v2/local/search/category.json",
                headers=headers, params=params, timeout=KAKAO_TIMEOUT_SEC
            )
        )
    
    # 병렬 실행
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    candidates = []
    
    for i, resp in enumerate(responses):
        if isinstance(resp, Exception) or resp.status_code != 200:
            continue
        data = resp.json()
        cat_code = KAKAO_CATEGORY_CODES[i]
        for doc in data.get("documents", []):
            doc["category_group_code"] = cat_code
            candidates.append(doc)
            
    return candidates


# -----------------------------
# [Async] 최적 POI 선정
# -----------------------------
async def choose_best_poi_for_turn_async(
    client: httpx.AsyncClient,
    simp_polyline: List[Tuple[float, float]],
    turn_dist: float, turn_lat: float, turn_lng: float
) -> Optional[Dict]:
    
    docs = await search_kakao_poi_candidates_async(client, turn_lat, turn_lng, KAKAO_POI_RADIUS_M)
    if not docs: return None

    best_poi = None
    best_score = None

    for doc in docs:
        try:
            name = shorten_poi_name(doc.get("place_name", ""))
            if not name: continue
            
            p_lat = float(doc["y"]); p_lng = float(doc["x"])
            cat = doc.get("category_group_code", "")
            
            dist_along, lateral = distance_along_polyline_to_point(simp_polyline, p_lat, p_lng)
            
            if lateral > POI_LATERAL_MAX_M: continue
            if dist_along < turn_dist - POI_SCAN_BEFORE_M: continue
            if dist_along > turn_dist + POI_SCAN_AFTER_M: continue
            
            relation = classify_poi_relation_to_turn(dist_along, turn_dist)
            score = (brand_priority(name), CATEGORY_RANK.get(cat, 5), abs(dist_along - turn_dist), lateral)
            
            if (best_score is None) or (score < best_score):
                best_score = score
                best_poi = {
                    "name": name, "lat": p_lat, "lng": p_lng,
                    "category_group_code": cat, "relation": relation,
                    "dist_along": dist_along, "lateral": lateral
                }
        except Exception:
            continue
    return best_poi


# -----------------------------
# [Async] 메인: 턴바이턴 빌더
# -----------------------------
async def build_turn_by_turn_async(
    polyline: List[Tuple[float, float]],
    km_requested: float,
) -> Tuple[List[Dict], Dict]:
    
    if len(polyline) < 2:
        return [], {"length_m": 0.0, "km_requested": km_requested, "estimated_time_min": 0.0, "event_count": 0}

    total_len = polyline_length_m(polyline)
    simp = simplify_polyline(polyline, MIN_DIST_SIMPLIFY)
    if len(simp) < 2: simp = polyline[:]

    # 1) Raw Turns 추출
    raw_turns = []
    cum_dist = 0.0
    last_turn_at = 0.0
    prev_bearing = None
    last_pt = simp[0]
    
    checkpoints_raw = []
    next_cp = CHECKPOINT_INTERVAL

    for i in range(1, len(simp)):
        cur_pt = simp[i]
        seg = haversine_m(*last_pt, *cur_pt)
        cum_dist += seg
        
        while cum_dist >= next_cp:
            checkpoints_raw.append({"dist_m": next_cp})
            next_cp += CHECKPOINT_INTERVAL
        
        if prev_bearing is not None:
            curr_bearing = bearing_deg(*last_pt, *cur_pt)
            if angle_diff_deg(prev_bearing, curr_bearing) >= ANGLE_TURN and (cum_dist - last_turn_at) >= MIN_DIST_TURN:
                ttype = _classify_turn(prev_bearing, curr_bearing)
                if ttype != "straight":
                    raw_turns.append({
                        "index": i, "lat": cur_pt[0], "lng": cur_pt[1],
                        "dist_m": cum_dist, "type": ttype
                    })
                    last_turn_at = cum_dist
            prev_bearing = curr_bearing
        else:
            prev_bearing = bearing_deg(*last_pt, *cur_pt)
        last_pt = cur_pt

    # 2) POI 병렬 조회
    async with httpx.AsyncClient() as client:
        tasks = [
            choose_best_poi_for_turn_async(client, simp, t["dist_m"], t["lat"], t["lng"])
            for t in raw_turns
        ]
        poi_results = await asyncio.gather(*tasks)

    for i, t in enumerate(raw_turns):
        poi = poi_results[i]
        if poi:
            t.update({
                "poi_name": poi["name"],
                "poi_relation": poi["relation"],
                "poi_dist_along": poi["dist_along"],
                "poi_lateral": poi["lateral"],
                "poi_category_group_code": poi["category_group_code"]
            })
        else:
            t["poi_name"] = None

    # 3) 이벤트 생성
    events = []
    events.append({"type": "start", "lat": simp[0][0], "lng": simp[0][1], "at_dist_m": 0.0, "instruction": "러닝을 시작합니다. 계속 직진하세요."})
    
    raw_turns.sort(key=lambda x: x["dist_m"])

    # 체크포인트 이벤트
    for cp in checkpoints_raw:
        d = cp["dist_m"]
        next_turn = next((t for t in raw_turns if t["dist_m"] >= d), None)
        if next_turn and (next_turn["dist_m"] - d) < CHECKPOINT_TURN_BUFFER:
            continue
        
        clat, clng = interpolate_point_along_polyline(simp, d)
        msg = f"지금까지 {int(d)}m 이동했습니다. "
        if next_turn:
            msg += f"다음은 {_turn_korean(next_turn['type'])}입니다. 계속 직진하세요."
        else:
            msg += "코스를 따라 계속 직진하세요."
        events.append({"type": "checkpoint", "lat": clat, "lng": clng, "at_dist_m": round(d, 1), "instruction": msg})

    # 턴 이벤트
    for t in raw_turns:
        dist = t["dist_m"]; ttype = t["type"]; kname = _turn_korean(ttype)
        pname = t.get("poi_name"); prel = t.get("poi_relation")

        # Pre-alert
        pre_d = dist - PRE_ALERT_DIST
        if pre_d > 0:
            plat, plng = interpolate_point_along_polyline(simp, pre_d)
            if pname:
                if prel == "before": msg = f"{int(PRE_ALERT_DIST)}m 앞 {pname} 지나서 {kname}입니다."
                elif prel == "at": msg = f"{int(PRE_ALERT_DIST)}m 앞 {pname}에서 {kname}입니다."
                else: msg = f"{int(PRE_ALERT_DIST)}m 앞 {kname}입니다. {kname} 후 {pname}이 나옵니다."
            else:
                msg = f"{int(PRE_ALERT_DIST)}m 앞에서 {kname}입니다."
            events.append({"type": f"pre_{ttype}", "lat": plat, "lng": plng, "at_dist_m": round(pre_d, 1), "instruction": msg + " 계속 직진하세요."})

        # Exec
        ex_d = max(0, dist - EXEC_ALERT_DIST)
        elat, elng = interpolate_point_along_polyline(simp, ex_d)
        if pname:
            if prel == "before": msg = f"이제 {pname} 지나서 {kname}하세요."
            elif prel == "at": msg = f"이제 {pname}에서 {kname}하세요."
            else: msg = f"이제 {kname}하세요. {kname} 후 {pname}이 나옵니다."
        else:
            msg = f"이제 {kname}하세요."
        events.append({"type": ttype, "lat": elat, "lng": elng, "at_dist_m": round(ex_d, 1), "instruction": msg})

        # After
        af_d = dist + AFTER_TURN_DIST
        if af_d <= total_len:
            alat, alng = interpolate_point_along_polyline(simp, af_d)
            events.append({"type": "after_turn", "lat": alat, "lng": alng, "at_dist_m": round(af_d, 1), "instruction": "잘하셨어요. 다음 안내까지 직진하세요."})

    events.sort(key=lambda e: e.get("at_dist_m", 0.0))
    
    est_time = (total_len / 1000.0) / (RUNNING_SPEED_KMH / 60.0)
    summary = {
        "length_m": round(total_len, 1),
        "km_requested": km_requested,
        "estimated_time_min": round(est_time, 1),
        "event_count": len(events)
    }
    return events, summary
