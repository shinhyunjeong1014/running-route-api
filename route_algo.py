import math
import os
import time
import logging
import json
from typing import List, Dict, Tuple, Optional, Any

import requests

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 기본 설정 (강화)
# -----------------------------

VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")
VALHALLA_TIMEOUT = float(os.environ.get("VALHALLA_TIMEOUT", "2.5"))
VALHALLA_MAX_RETRY = int(os.environ.get("VALHALLA_MAX_RETRY", "1"))

# [핵심] 카카오 API 설정
KAKAO_API_KEY = "dc3686309f8af498d7c62bed0321ee64"
KAKAO_ROUTE_URL = "https://apis-navi.kakaomobility.com/v1/directions"

RUNNING_SPEED_KMH = 8.0  
GLOBAL_TIMEOUT_S = 10.0 
MAX_TOTAL_CALLS = 30 
MAX_LENGTH_ERROR_M = 99.0
MAX_BEST_ROUTES_TO_TEST = 5 
MAX_ROUTES_TO_PROCESS = 10 # Valhalla/Kakao 복귀 경로 후보 수

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

def valhalla_route(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    is_shrink_attempt: bool = False
) -> List[Tuple[float, float]]:
    lat1, lon1 = p1; lat2, lon2 = p2
    last_error: Optional[Exception] = None
    
    costing_options = {
        "pedestrian": {
            "avoid_steps": 1.0, 
            "service_penalty": 1000, 
            "use_hills": 0.0,
            "use_ferry": 0.0,
            "track_type_penalty": 50, 
            "private_road_penalty": 10000,
        }
    }
    
    for attempt in range(VALHALLA_MAX_RETRY):
        try:
            payload = {
                "locations": [{"lat": lat1, "lon": lon1, "type": "break"}, {"lat": lat2, "lon": lon2, "type": "break"}],
                "costing": "pedestrian",
                "costing_options": costing_options
            }
            # 이 로직에서는 avoid_polygons를 사용하지 않습니다 (성공률 저하 방지)

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

def kakao_walk_route(p1: Tuple[float, float], p2: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
    """카카오 길찾기 API (도보)를 호출하여 경로 폴리라인을 반환"""
    if not KAKAO_API_KEY:
        logger.error("[Kakao API] KAKAO_API_KEY not configured.")
        return None
    
    lon1, lat1 = p1[::-1]; lon2, lat2 = p2[::-1]

    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"origin": f"{lon1},{lat1}", "destination": f"{lon2},{lat2}", "waypoints": "", "priority": "RECOMMEND", "car_model": "walk"}

    try:
        resp = requests.get(KAKAO_ROUTE_URL, params=params, headers=headers, timeout=VALHALLA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("routes") and data["routes"][0]["result_code"] == 0:
            coords = []
            for route in data["routes"]:
                for section in route["sections"]:
                    for road in section.get("roads", []):
                        vertices = road.get("vertexes", [])
                        
                        for i in range(0, len(vertices), 2): 
                            if i + 1 < len(vertices):
                                lon = vertices[i]
                                lat = vertices[i+1]
                                coords.append((lat, lon))
            
            if coords and len(coords) >= 2:
                if coords[0] != p1: coords.insert(0, p1)
                if coords[-1] != p2: coords.append(p2)
                return coords
        
    except Exception as e:
        logger.error("[Kakao API] Request failed (Parsing or Network): %s", e)
        return None
    return None

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
                logger.error(f"[Valhalla Decode] Sanity check failed: ({current_lat}, {current_lng})")
                return []
            coords.append((current_lat, current_lng))
    except IndexError:
        logger.error("[Valhalla Decode] Unexpected end of polyline string.")
        return []
    return coords


# -----------------------------
# 루프 품질 평가 / 단축 재연결 로직
# -----------------------------

def _loop_roundness(points: List[Tuple[float, float]]) -> float:
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
    length_m = polyline_length_m(points)
    if length_m <= 0.0:
        return float("inf"), {"len": 0.0, "err": target_m, "roundness": 0.0, "score": float("inf")}
    err = abs(length_m - target_m); roundness = _loop_roundness(points)
    score = err + (1.0 - roundness) * 0.3 * target_m
    length_ok = True 
    return score, {"len": length_m, "err": err, "roundness": roundness, "score": score, "length_ok": length_ok}

def _is_path_safe(points: List[Tuple[float, float]]) -> bool:
    """ 안전성 기준을 제거했으므로, 이 함수는 항상 True를 반환합니다. """
    return True 

def _try_shrink_path_kakao(
    current_route: List[Tuple[float, float]],
    target_m: float,
    valhalla_calls: int,
    start_time: float,
    global_timeout: float,
) -> Tuple[Optional[List[Tuple[float, float]]], int]:
    
    current_len = polyline_length_m(current_route)
    error_m = current_len - target_m
    
    if time.time() - start_time >= global_timeout:
        return None, valhalla_calls

    target_reduction = error_m 
    
    pts = current_route
    idx_a = max(1, int(len(pts) * 0.40))
    idx_b = min(len(pts) - 2, int(len(pts) * 0.60))
    
    if idx_a < idx_b:
        p_a = pts[idx_a]; p_b = pts[idx_b]
        
        reconnect_seg = kakao_walk_route(p_a, p_b)
        
        if reconnect_seg and len(reconnect_seg) >= 2:
            seg_len_original = polyline_length_m(pts[idx_a : idx_b + 1])
            seg_len_new = polyline_length_m(reconnect_seg)
            reduction = seg_len_original - seg_len_new

            if reduction > target_reduction * 0.5 and reduction > 0:
                new_route = pts[:idx_a] + reconnect_seg + pts[idx_b+1:]
                
                final_len = polyline_length_m(new_route)
                
                if abs(final_len - target_m) <= MAX_LENGTH_ERROR_M:
                    return new_route, valhalla_calls

    return None, valhalla_calls

def _calculate_overlap_penalty(seg_out: List[Tuple[float, float]], seg_back: List[Tuple[float, float]]) -> float:
    """
    복귀 경로(seg_back)가 나가는 경로(seg_out)와 공간적으로 겹치는 정도를 측정하여 페널티를 부과합니다.
    """
    if not seg_out or not seg_back: return 0.0

    overlap_count = 0
    OVERLAP_THRESHOLD_DEG = 0.0002 # 약 20m 근접성
    
    for lat_c, lon_c in seg_back:
        is_close = False
        for lat_a, lon_a in seg_out:
            if abs(lat_c - lat_a) < OVERLAP_THRESHOLD_DEG and abs(lon_c - lon_a) < OVERLAP_THRESHOLD_DEG:
                is_close = True
                break
        if is_close:
            overlap_count += 1

    seg_back_len = len(seg_back)
    if seg_back_len > 0 and overlap_count / seg_back_len > 0.1: # 10% 이상 겹치면 페널티
        overlap_ratio = overlap_count / seg_back_len
        return overlap_ratio * 200.0 # 200m 상당의 페널티 부과
        
    return 0.0


# -----------------------------
# Area Loop 생성 (Two-Segment Hybrid)
# -----------------------------

def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
) -> Tuple[List[Tuple[float, float]], Dict]:
    """[최종] 목표 거리(km)를 위해 Two-Segment 구조를 사용하며, 겹침 페널티를 적용합니다."""
    
    start_time = time.time()
    
    target_m = max(300.0, km * 1000.0) 
    km_requested = km
    start = (lat, lng)

    if time.time() - start_time >= GLOBAL_TIMEOUT_S:
         return [start], {"len": 0.0, "err": target_m, "success": False, "used_fallback": False, "valhalla_calls": 0, "time_s": 0.0, "message": "경로 생성 요청이 시작하자마자 시간 제한(10초)을 초과했습니다."}

    SEGMENT_LEN = target_m / 2.0 # Half distance for two segments
    R_ideal = target_m / (2.0 * math.pi)
    
    # R 제약 완화 (100m까지 낮춰 탐색 공간 최대화)
    R_MIN = max(100.0, min(R_ideal * 0.3, 200.0))
    R_SMALL = max(200.0, min(R_ideal * 0.6, 400.0))
    R_MEDIUM = max(400.0, min(R_ideal * 1.0, 700.0))
    R_LARGE = max(700.0, min(R_ideal * 1.3, 1100.0))
    R_XLARGE = max(1100.0, min(R_ideal * 1.6, 1800.0))
    
    radii = list(sorted(list(set([R_MIN, R_SMALL, R_MEDIUM, R_LARGE, R_XLARGE]))))
    bearings = [0, 45, 90, 135, 180, 225, 270, 315] 

    candidate_routes = []
    valhalla_calls = 0
    total_routes_checked = 0

    # 1. Valhalla 탐색 (전수 조사: Start -> Comeback)
    for R in radii:
        if valhalla_calls + 2 > MAX_TOTAL_CALLS: break
        if time.time() - start_time >= GLOBAL_TIMEOUT_S: break

        for br in bearings:
            if valhalla_calls + 2 > MAX_TOTAL_CALLS: break
            if time.time() - start_time >= GLOBAL_TIMEOUT_S: break

            via_a = project_point(lat, lng, R, br)
            
            # [수정] 1차 경로 생성: Start -> Comeback (Far Away)
            seg_out = valhalla_route(start, via_a); valhalla_calls += 1
            if not seg_out or len(seg_out) < 2: continue
            
            # Comeback Point 설정 (seg_out의 끝점)
            comback_point = seg_out[-1]
            
            # 2. 2차 경로 생성 후보 확보 (Comeback -> Start)
            
            back_segments = []
            
            # 2.1 Valhalla 복귀 경로 (2회 시도)
            if valhalla_calls + 1 <= MAX_TOTAL_CALLS:
                seg_back_v = valhalla_route(comback_point, start)
                valhalla_calls += 1
                if seg_back_v and len(seg_back_v) >= 2: back_segments.append({"seg": seg_back_v, "source": "Valhalla"})
            
            # 2.2 카카오 복귀 경로 (1회 시도)
            seg_back_k = kakao_walk_route(comback_point, start)
            if seg_back_k and len(seg_back_k) >= 2: back_segments.append({"seg": seg_back_k, "source": "Kakao"})

            # 3. 최종 루프 구성 및 페널티 적용
            for back_seg_data in back_segments:
                seg_back = back_seg_data["seg"]
                
                # 겹침 페널티 계산 (핵심)
                overlap_penalty = _calculate_overlap_penalty(seg_out, seg_back)
                
                total_route = seg_out + seg_back[1:] # 겹치는 Comeback Point 제거
                if total_route and total_route[0] != start: total_route.insert(0, start)
                if total_route and total_route[-1] != start: total_route.append(start)
                temp_pts = [total_route[0]]; [temp_pts.append(p) for p in total_route[1:] if p != temp_pts[-1]]; total_route = temp_pts
                
                # 최종 점수 계산 (Roundness와 길이 오차 + 겹침 페널티)
                score_base, local_meta = _score_loop(total_route, target_m)
                total_score = score_base + overlap_penalty 
                
                if polyline_length_m(total_route) > 0:
                    candidate_routes.append({
                        "route": total_route, 
                        "valhalla_score": total_score, # 페널티가 포함된 점수
                    })
                    total_routes_checked += 1

        if valhalla_calls + 3 > MAX_TOTAL_CALLS: break

    # -----------------------------
    # 4. 모든 후보 경로 후처리 (카카오 단축 시도)
    # -----------------------------
    
    final_validated_routes = []
    candidate_routes.sort(key=lambda x: x["valhalla_score"])
    
    for i, candidate in enumerate(candidate_routes[:MAX_BEST_ROUTES_TO_TEST]): 
        
        if time.time() - start_time >= GLOBAL_TIMEOUT_S: break

        current_route = candidate['route']
        final_len = polyline_length_m(current_route)
        
        # 1. 이미 ±99m 이내인 경우 (단축 불필요)
        if abs(final_len - target_m) <= MAX_LENGTH_ERROR_M:
            final_validated_routes.append({"route": current_route, "score": _score_loop(current_route, target_m)[0]})
            continue
            
        # 2. 길이가 99m 초과하면 카카오 단축 시도
        if final_len > target_m + MAX_LENGTH_ERROR_M:
            shrunken_route, valhalla_calls = _try_shrink_path_kakao(
                current_route, target_m, valhalla_calls, start_time, GLOBAL_TIMEOUT_S
            )

            if shrunken_route:
                final_score = _score_loop(shrunken_route, target_m)[0]
                final_validated_routes.append({
                    "route": shrunken_route, 
                    "score": final_score
                })

    # -----------------------------
    # 5. 최종 베스트 경로 선택 (절대 반환 보장)
    # -----------------------------
    
    best_final_route = None
    
    if final_validated_routes:
        # A. ±99m를 만족하는 경로 중 최적 경로 선택
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
