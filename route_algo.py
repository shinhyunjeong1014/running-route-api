import math
import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Any

import requests

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 기본 설정 (PSP2 및 길이 제어)
# -----------------------------

VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")
VALHALLA_TIMEOUT = float(os.environ.get("VALHALLA_TIMEOUT", "2.5"))
VALHALLA_MAX_RETRY = int(os.environ.get("VALHALLA_MAX_RETRY", "2")) 

KAKAO_API_KEY = "dc3686309f8af498d7c62bed0321ee64"
KAKAO_ROUTE_URL = "https://apis-navi.kakaomobility.com/v1/directions"

RUNNING_SPEED_KMH = 8.0  
GLOBAL_TIMEOUT_S = 10.0 
MAX_TOTAL_CALLS = 30 
MAX_LENGTH_ERROR_M = 99.0
MAX_BEST_ROUTES_TO_TEST = 5 
MAX_ROUTES_TO_PROCESS = 10 

# PSP2 알고리즘의 길이 오차 파라미터 (epsilon)
# 논문에 따르면, L의 10% (epsilon=0.1) 정도가 합리적이나, 저희는 +/- 99m를 사용합니다.
# 따라서, L/3의 오차 범위는 L의 99m를 초과하지 않도록 설정해야 합니다.
PSP2_SEGMENT_ERROR_FACTOR = 0.05 # 5% 오차 (L/3의 5%는 L의 1.66%에 해당)


# -----------------------------
# 거리 / 기하 유틸
# -----------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 사이의 대략적인 거리(m)."""
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = p2 - p1; dl = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def polyline_length_m(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2: return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        total += haversine_m(lat1, lon1, lat2, lon2)
    return total
# ... (bearing_deg, project_point 함수는 생략됨)
def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2); y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0
def project_point(lat: float, lon: float, distance_m: float, bearing_deg_: float) -> Tuple[float, float]:
    R = 6371000.0; br = math.radians(bearing_deg_)
    phi1 = math.radians(lat); lam1 = math.radians(lon)
    phi2 = math.asin(math.sin(phi1) * math.cos(distance_m / R) + math.cos(phi1) * math.sin(distance_m / R) * math.cos(br))
    lam2 = lam1 + math.atan2(math.sin(br) * math.sin(distance_m / R) * math.cos(phi1), math.cos(distance_m / R) - math.sin(phi1) * math.sin(phi2))
    return (math.degrees(phi2), (math.degrees(lam2) + 540.0) % 360.0 - 180.0)


# -----------------------------
# Valhalla/Kakao API 호출
# -----------------------------

def _get_bounding_box_polygon(points: List[Tuple[float, float]], buffer_deg: float = 0.00001) -> Optional[List[Tuple[float, float]]]:
    if not points: return None
    min_lat = min(p[0] for p in points); max_lat = max(p[0] for p in points)
    min_lon = min(p[1] for p in points); max_lon = max(p[1] for p in points)
    if haversine_m(min_lat, min_lon, max_lat, max_lon) < 20: return None 
    buf = buffer_deg
    return [(min_lat - buf, min_lon - buf), (max_lat + buf, min_lon - buf), (max_lat + buf, max_lon + buf), (min_lat - buf, max_lon + buf), (min_lat - buf, min_lon - buf)]

def valhalla_route(
    p1: Tuple[float, float], p2: Tuple[float, float], avoid_polygons: Optional[List[List[Tuple[float, float]]]] = None, is_shrink_attempt: bool = False
) -> List[Tuple[float, float]]:
    lat1, lon1 = p1; lat2, lon2 = p2
    last_error: Optional[Exception] = None
    
    # [논문 기반] 안전/도보 선호 Costing Options
    costing_options = {
        "pedestrian": {
            "avoid_steps": 1.0, "service_penalty": 1000, "use_hills": 0.0, "use_ferry": 0.0,
            "track_type_penalty": 0, "private_road_penalty": 100000,
            "sidewalk_preference": 1.0, "alley_preference": -1.0, "max_road_class": 0.5,
            "length_penalty": 0.0 
        }
    }
    
    for attempt in range(VALHALLA_MAX_RETRY):
        try:
            payload = {"locations": [{"lat": lat1, "lon": lon1, "type": "break"}, {"lat": lat2, "lon": lon2, "type": "break"}],
                "costing": "pedestrian", "costing_options": costing_options}
            if avoid_polygons:
                valhalla_polys = []; [valhalla_polys.append([[lon, lat] for lat, lon in poly]) for poly in avoid_polygons]
                payload["avoid_polygons"] = valhalla_polys

            resp = requests.post(VALHALLA_URL, json=payload, timeout=VALHALLA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            shape = data["trip"]["legs"][0]["shape"]
            return _decode_polyline(shape)
        except Exception as e:
            last_error = e
            logger.warning("[Valhalla] attempt %d failed for %s -> %s: %s", attempt + 1, p1, p2, e)
    logger.error("[Valhalla] all attempts failed for %s -> %s: %s", p1, p2, last_error)
    return []

# [KAKAO API는 논문 기반 설계에서 사용되지 않음 - 트레이드오프]
# def kakao_walk_route(...): ...
# ... (다른 필수 헬퍼 함수 정의)
def _decode_polyline(shape: str) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []; lat = 0; lng = 0; idx = 0; precision = 1e6
    try:
        while idx < len(shape):
            shift = 0; result = 0
            while True:
                b = ord(shape[idx]) - 63; idx += 1; result |= (b & 0x1F) << shift; shift += 5
                if b < 0x20: break
            dlat = ~(result >> 1) if (result & 1) else (result >> 1); lat += dlat
            shift = 0; result = 0
            while True:
                b = ord(shape[idx]) - 63; idx += 1; result |= (b & 0x1F) << shift; shift += 5
                if b < 0x20: break
            dlng = ~(result >> 1) if (result & 1) else (result >> 1); lng += dlng
            current_lat = lat / precision; current_lng = lng / precision
            if not (-90.0 <= current_lat <= 90.0 and -180.0 <= current_lng <= 180.0):
                logger.error(f"[Valhalla Decode] Sanity check failed: ({current_lat}, {current_lng})"); return []
            coords.append((current_lat, current_lng))
    except IndexError:
        logger.error("[Valhalla Decode] Unexpected end of polyline string."); return []
    return coords


# -----------------------------
# 루프 품질 평가 / 최종 길이 조정 로직
# -----------------------------

def _loop_roundness(points: List[Tuple[float, float]]) -> float:
    # ... (로직 유지)
    if len(points) < 4: return 0.0
    xs = [p[1] for p in points]; ys = [p[0] for p in points]
    cx = sum(xs) / len(xs); cy = sum(ys) / len(ys)
    dists = [haversine_m(cy, cx, lat, lon) for lat, lon in points]
    mean_r = sum(dists) / len(dists)
    if mean_r <= 0: return 0.0
    var = sum((d - mean_r) ** 2 for d in dists) / len(dists)
    score = 1.0 / (1.0 + var / (mean_r * mean_r + 1e-6))
    return max(0.0, min(1.0, score))

def _score_loop(
    points: List[Tuple[float, float]], target_m: float
) -> Tuple[float, Dict]:
    # ... (로직 유지)
    length_m = polyline_length_m(points)
    if length_m <= 0.0: return float("inf"), {"len": 0.0, "err": target_m, "roundness": 0.0, "score": float("inf")}
    err = abs(length_m - target_m); roundness = _loop_roundness(points)
    score = err + (1.0 - roundness) * 0.3 * target_m
    length_ok = True 
    return score, {"len": length_m, "err": err, "roundness": roundness, "score": score, "length_ok": length_ok}

def _calculate_overlap_penalty(seg_out: List[Tuple[float, float]], seg_back: List[Tuple[float, float]]) -> float:
    """ 경로 중복 페널티 계산 (논문 기반) """
    if not seg_out or not seg_back: return 0.0
    overlap_count = 0
    OVERLAP_THRESHOLD_DEG = 0.0002 
    for lat_c, lon_c in seg_back:
        is_close = False
        for lat_a, lon_a in seg_out:
            if abs(lat_c - lat_a) < OVERLAP_THRESHOLD_DEG and abs(lon_c - lon_a) < OVERLAP_THRESHOLD_DEG:
                is_close = True; break
        if is_close: overlap_count += 1
    seg_back_len = len(seg_back)
    if seg_back_len > 0 and overlap_count / seg_back_len > 0.1:
        overlap_ratio = overlap_count / seg_back_len
        return overlap_ratio * 1000.0
    return 0.0

def _trim_path_to_length(route: List[Tuple[float, float]], target_m: float) -> List[Tuple[float, float]]:
    """경로가 길 때, 끝 부분을 강제로 잘라 목표 길이(±99m)에 맞춥니다."""
    current_len = polyline_length_m(route)
    if current_len <= target_m + MAX_LENGTH_ERROR_M: return route

    required_trim = current_len - target_m
    trimmed_route = route[:]
    reversed_route = trimmed_route[::-1]
    current_dist_removed = 0.0
    
    for i in range(len(reversed_route) - 1):
        p1 = reversed_route[i]
        seg_len = haversine_m(p1[0], p1[1], reversed_route[i+1][0], reversed_route[i+1][1])
        
        if current_dist_removed + seg_len >= required_trim:
            final_index = len(route) - (i + 1)
            trimmed_route = route[:final_index]
            
            if trimmed_route and trimmed_route[-1] != route[0]:
                trimmed_route.append(route[0])
            
            return trimmed_route
        
        current_dist_removed += seg_len
        
    return route 

# -----------------------------
# Area Loop 생성 (PSP2 원리 기반)
# -----------------------------

def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
) -> Tuple[List[Tuple[float, float]], Dict]:
    """[최종] 목표 거리(km)를 위해 Two-Segment 구조를 사용하며, Trimming으로 길이 정밀도를 강제합니다."""
    
    start_time = time.time()
    
    target_m = max(300.0, km * 1000.0) 
    km_requested = km
    start = (lat, lng)

    if time.time() - start_time >= GLOBAL_TIMEOUT_S:
         return [start], {"len": 0.0, "err": target_m, "success": False, "used_fallback": False, "valhalla_calls": 0, "time_s": 0.0, "message": "경로 생성 요청이 시작하자마자 시간 제한(10초)을 초과했습니다.", "routes_checked": 0}

    # PSP2 원리: L/3 길이 제약으로 Via 지점 탐색
    L_THIRD = target_m / 3.0
    
    # R 제약 완화 (100m까지 낮춰 탐색 공간 최대화)
    R_MIN = max(100.0, L_THIRD * 0.3)
    R_SMALL = max(200.0, L_THIRD * 0.6)
    R_MEDIUM = max(400.0, L_THIRD * 0.9)
    R_LARGE = max(700.0, L_THIRD * 1.2)
    R_XLARGE = max(1100.0, L_THIRD * 1.5)
    
    radii = list(sorted(list(set([R_MIN, R_SMALL, R_MEDIUM, R_LARGE, R_XLARGE]))))
    bearings = [0, 45, 90, 135, 180, 225, 270, 315] 

    candidate_routes = []
    valhalla_calls = 0
    total_routes_checked = 0

    # 1. Valhalla 탐색 (전수 조사)
    for R in radii:
        if valhalla_calls + 2 > MAX_TOTAL_CALLS: break
        if time.time() - start_time >= GLOBAL_TIMEOUT_S: break

        for br in bearings:
            if valhalla_calls + 2 > MAX_TOTAL_CALLS: break
            if time.time() - start_time >= GLOBAL_TIMEOUT_S: break

            via_a = project_point(lat, lng, R, br)
            
            # 1) Seg A: 출발 → Via A (Out Segment)
            seg_out = valhalla_route(start, via_a); valhalla_calls += 1
            if not seg_out or len(seg_out) < 2: continue
            
            comback_point = seg_out[-1]
            
            # 2. 2차 경로 생성 후보 확보 (Comeback → Start)
            
            back_segments = []
            
            # 2.1 Valhalla 복귀 경로 (2회 시도)
            if valhalla_calls + 1 <= MAX_TOTAL_CALLS:
                seg_back_v = valhalla_route(comback_point, start)
                valhalla_calls += 1
                if seg_back_v and len(seg_back_v) >= 2: back_segments.append({"seg": seg_back_v, "source": "Valhalla"})
            
            # 2.2 카카오 복귀 경로는 사용하지 않음 (순수 Valhalla)
            
            # 3. 최종 루프 구성 및 페널티 적용
            for back_seg_data in back_segments:
                seg_back = back_seg_data["seg"]
                
                # 겹침 페널티 계산 (핵심)
                overlap_penalty = _calculate_overlap_penalty(seg_out, seg_back)
                
                if overlap_penalty > 300.0: continue

                total_route = seg_out + seg_back[1:] 
                if total_route and total_route[0] != start: total_route.insert(0, start)
                if total_route and total_route[-1] != start: total_route.append(start)
                temp_pts = [total_route[0]]; [temp_pts.append(p) for p in total_route[1:] if p != temp_pts[-1]]; total_route = temp_pts
                
                score_base, local_meta = _score_loop(total_route, target_m)
                total_score = score_base + overlap_penalty # 최종 점수에 겹침 페널티 부과
                
                # [핵심] 안전성 필터 통과한 경로만 저장 (안전성 필터는 복구됨)
                if _is_path_safe(total_route) and polyline_length_m(total_route) > 0:
                    candidate_routes.append({
                        "route": total_route, 
                        "valhalla_score": total_score, # 페널티가 포함된 점수
                    })
                    total_routes_checked += 1

        if valhalla_calls + 2 > MAX_TOTAL_CALLS: break

    # -----------------------------
    # 4. 모든 후보 경로 후처리 (최종 Trimming)
    # -----------------------------
    
    final_validated_routes = []
    candidate_routes.sort(key=lambda x: x["valhalla_score"])
    
    # [핵심] 모든 후보 경로에 대해 Trimming 시도
    for i, candidate in enumerate(candidate_routes): 
        
        if time.time() - start_time >= GLOBAL_TIMEOUT_S: break
        
        # 4.1. 안전성 필터를 통과한 경로에 대해서만 길이 조정 시도
        current_route = candidate['route']
        final_len = polyline_length_m(current_route)
        
        # 1. 이미 ±99m 이내인 경우
        if abs(final_len - target_m) <= MAX_LENGTH_ERROR_M:
            final_validated_routes.append({"route": current_route, "score": _score_loop(current_route, target_m)[0]})
            continue
            
        # 2. 길이가 99m 초과하면 강제 Trimming 적용
        if final_len > target_m + MAX_LENGTH_ERROR_M:
            
            trimmed_route = _trim_path_to_length(current_route, target_m)
            
            # Trimming 후에도 99m 이내를 만족하면 최종 후보에 추가 (Trimming 성공)
            if trimmed_route and abs(polyline_length_m(trimmed_route) - target_m) <= MAX_LENGTH_ERROR_M:
                final_score = _score_loop(trimmed_route, target_m)[0]
                final_validated_routes.append({
                    "route": trimmed_route, 
                    "score": final_score
                })

    # -----------------------------
    # 5. 최종 베스트 경로 선택 (절대 반환 보장)
    # -----------------------------
    
    best_final_route = None
    
    if final_validated_routes:
        final_validated_routes.sort(key=lambda x: x["score"])
        best_final_route = final_validated_routes[0]["route"]
        
    elif candidate_routes:
        # B. ±99m를 만족하는 경로가 없으면, 모든 원본 후보 중 '가장 인접한' 경로를 선택
        min_error = float("inf")
        most_adjacent_route = None

        for candidate in candidate_routes:
            length = polyline_length_m(candidate["route"])
            error = abs(length - target_m)
            
            if error < min_error:
                min_error = error
                most_adjacent_route = candidate["route"]
                
        best_final_route = most_adjacent_route
    
    # -----------------------------
    # 6. 결과 반환 (성공/최인접 경로 반환 보장)
    # -----------------------------
    
    if best_final_route:
        final_len = polyline_length_m(best_final_route)
        is_perfect = abs(final_len - target_m) <= MAX_LENGTH_ERROR_M
        
        meta = {
            "len": final_len, "err": abs(final_len - target_m), "roundness": _loop_roundness(best_final_route), 
            "success": is_perfect,
            "used_fallback": False, 
            "valhalla_calls": valhalla_calls, "time_s": round(time.time() - start_time, 2),
            "message": "최적의 경로가 도출되었습니다." if is_perfect else "요청 오차(±99m)를 초과하지만, 가장 인접한 경로를 반환합니다.",
            "length_ok": is_perfect,
            "routes_checked": total_routes_checked,
            "routes_processed": len(candidate_routes),
            "routes_validated": len(final_validated_routes)
        }
        
        return best_final_route, meta

    # -----------------------------
    # 7. 최종 실패 (경로 후보가 0개)
    # -----------------------------
    return [start], {
        "len": 0.0, "err": target_m, "success": False, "used_fallback": False, 
        "valhalla_calls": valhalla_calls, "time_s": round(time.time() - start_time, 2),
        "message": f"탐색 결과, 유효한 경로 후보를 찾을 수 없습니다. (Valhalla 통신 불가 또는 지리적 단절)",
        "routes_checked": total_routes_checked,
    }
