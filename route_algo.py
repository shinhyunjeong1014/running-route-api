import math
import os
import time
import logging
import json
from typing import List, Dict, Tuple, Optional

import requests

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 기본 설정 (강화)
# -----------------------------

VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")
VALHALLA_TIMEOUT = float(os.environ.get("VALHALLA_TIMEOUT", "2.5"))
VALHALLA_MAX_RETRY = int(os.environ.get("VALHALLA_MAX_RETRY", "1"))

# [핵심] 카카오 API 설정 (제공해주신 키 사용)
KAKAO_API_KEY = "dc3686309f8af498d7c62bed0321ee64"
KAKAO_ROUTE_URL = "https://apis-navi.kakaomobility.com/v1/directions"

RUNNING_SPEED_KMH = 8.0  
MAX_TOTAL_CALLS = 14 
GLOBAL_TIMEOUT_S = 5.0 
MAX_LENGTH_ERROR_M = 99.0

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
        }
    }
    
    for attempt in range(VALHALLA_MAX_RETRY):
        try:
            payload = {
                "locations": [{"lat": lat1, "lon": lon1, "type": "break"}, {"lat": lat2, "lon": lon2, "type": "break"}],
                "costing": "pedestrian",
                "costing_options": costing_options
            }
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
            for section in data["routes"][0]["sections"]:
                # 카카오 API는 guide가 아닌 road_details에서 좌표를 제공할 수 있으므로,
                # 여기서는 가이드 지점만 사용하는 로직을 유지
                for guide in section["guides"]:
                    point = guide["point"].split(",")
                    if len(point) == 2:
                        coords.append((float(point[1]), float(point[0]))) 
            
            if coords and len(coords) >= 2:
                # 카카오 API는 Lon, Lat 순서로 반환하므로, Lat, Lon으로 변환되어 저장됨
                return coords
        
    except Exception as e:
        logger.error("[Kakao API] Request failed: %s", e)
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


def _try_shrink_path_kakao(
    current_route: List[Tuple[float, float]],
    target_m: float,
    valhalla_calls: int,
    start_time: float,
    global_timeout: float,
) -> Tuple[Optional[List[Tuple[float, float]]], int]:
    
    current_len = polyline_length_m(current_route)
    error_m = current_len - target_m
    
    # 단축 시도 횟수를 1회로 제한
    if time.time() - start_time >= global_timeout:
        return None, valhalla_calls

    target_reduction = error_m 
    
    # --- 단축 시도 1: 경로 중앙부에서 단축 시도 ---
    pts = current_route
    
    # A와 B 지점 설정: 중앙부에서 40% ~ 60% 지점을 후보로 사용
    idx_a = max(1, int(len(pts) * 0.40))
    idx_b = min(len(pts) - 2, int(len(pts) * 0.60))
    
    if idx_a < idx_b:
        p_a = pts[idx_a]
        p_b = pts[idx_b]
        
        # 1. 재연결 경로 요청 (카카오 API 호출)
        reconnect_seg = kakao_walk_route(p_a, p_b)
        
        # 카카오 호출은 Valhalla 호출 횟수에 포함시키지 않음
        
        if reconnect_seg and len(reconnect_seg) >= 2:
            seg_len_original = polyline_length_m(pts[idx_a : idx_b + 1])
            seg_len_new = polyline_length_m(reconnect_seg)
            reduction = seg_len_original - seg_len_new

            # 2. 단축에 성공했고, 목표 감축량에 근접하는지 확인 (최소 50% 감축)
            if reduction > target_reduction * 0.5 and reduction > 0:
                new_route = pts[:idx_a] + reconnect_seg + pts[idx_b+1:]
                
                # 최종 길이 검증
                final_len = polyline_length_m(new_route)
                
                if abs(final_len - target_m) <= MAX_LENGTH_ERROR_M:
                    return new_route, valhalla_calls

    return None, valhalla_calls


# -----------------------------
# Area Loop 생성 (삼각형 폐쇄 루프 방식)
# -----------------------------

def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
) -> Tuple[List[Tuple[float, float]], Dict]:
    """목표 거리(km) 근처의 '닫힌 러닝 루프'를 생성한다. (길이/안전성 강화 버전)"""
    
    start_time = time.time()
    
    target_m = max(300.0, km * 1000.0) 
    km_requested = km
    start = (lat, lng)

    if time.time() - start_time >= GLOBAL_TIMEOUT_S:
         return [start], {"len": 0.0, "err": target_m, "success": False, "used_fallback": False, "valhalla_calls": 0, "time_s": 0.0, "message": "경로 생성 요청이 시작하자마자 시간 제한(5초)을 초과했습니다."}

    SEGMENT_LEN = target_m / 3.0
    R_ideal = target_m / (2.0 * math.pi)
    
    # 골목길 회피 기준 유지 (R_MIN 450m 이상)
    R_MIN = max(450.0, min(R_ideal * 0.7, 500.0))
    R_SMALL = max(500.0, min(R_ideal * 0.9, 700.0))
    R_MEDIUM = max(700.0, min(R_ideal * 1.1, 1000.0))
    R_LARGE = max(900.0, min(R_ideal * 1.3, 1300.0))
    R_XLARGE = max(1100.0, min(R_ideal * 1.5, 1600.0))
    
    radii = list(sorted(list(set([R_MIN, R_SMALL, R_MEDIUM, R_LARGE, R_XLARGE]))))
    bearings = [0, 90, 180, 270] 

    best_route: List[Tuple[float, float]] = []; best_meta: Dict = {}; best_score = float("inf")
    valhalla_calls = 0

    # 1. 5단계 반경 + 4방위 테스트 (최대 12회 호출)
    for R in radii:
        if valhalla_calls + 3 > MAX_TOTAL_CALLS: 
            break
        if time.time() - start_time >= GLOBAL_TIMEOUT_S: 
            break

        for br in bearings:
            if valhalla_calls + 3 > MAX_TOTAL_CALLS: 
                break
            if time.time() - start_time >= GLOBAL_TIMEOUT_S: 
                break

            via_a = project_point(lat, lng, R, br)
            seg_dist = max(50.0, SEGMENT_LEN) 
            via_b = project_point(*via_a, seg_dist, (br + 120.0) % 360.0) 
            
            # 1) Seg A: 출발 → Via A
            seg_a = valhalla_route(start, via_a); valhalla_calls += 1
            if not seg_a or len(seg_a) < 2: continue

            # 2) Seg B: Via A → Via B
            if valhalla_calls + 2 > MAX_TOTAL_CALLS: break
            if time.time() - start_time >= GLOBAL_TIMEOUT_S: break
            seg_b = valhalla_route(seg_a[-1], via_b); valhalla_calls += 1
            if not seg_b or len(seg_b) < 2: continue

            # 3) Seg C: Via B → 출발
            if valhalla_calls + 1 > MAX_TOTAL_CALLS: break
            if time.time() - start_time >= GLOBAL_TIMEOUT_S: break
            seg_c = valhalla_route(seg_b[-1], start); valhalla_calls += 1
            if not seg_c or len(seg_c) < 2: continue

            loop_pts = seg_a + seg_b[1:] + seg_c[1:]
            if loop_pts and loop_pts[0] != start: loop_pts[0] = start
            if loop_pts and loop_pts[-1] != start: loop_pts[-1] = start
            temp_pts = [loop_pts[0]]; [temp_pts.append(p) for p in loop_pts[1:] if p != temp_pts[-1]]; loop_pts = temp_pts
            
            score, local_meta = _score_loop(loop_pts, target_m)
            
            final_len = polyline_length_m(loop_pts)
            if abs(final_len - target_m) <= MAX_LENGTH_ERROR_M and score < best_score:
                best_score = score; best_route = loop_pts; best_meta = local_meta

        if valhalla_calls + 3 > MAX_TOTAL_CALLS: break

    # -----------------------------
    # 2. 결과 정리 (성공 케이스)
    # -----------------------------
    if best_route:
        final_len = polyline_length_m(best_route)
        
        # [핵심] 길이가 99m 초과 시 카카오 단축 로직 시도 (최후의 수단)
        if abs(final_len - target_m) > MAX_LENGTH_ERROR_M and final_len > target_m:
            logger.info("[Loop Gen] Path too long. Attempting Kakao shrink...")
            shrunken_route, valhalla_calls = _try_shrink_path_kakao(
                best_route, target_m, valhalla_calls, start_time, GLOBAL_TIMEOUT_S
            )

            if shrunken_route:
                best_route = shrunken_route
                final_len = polyline_length_m(best_route) 
                
                best_meta.update({
                    "len": final_len, "err": abs(final_len - target_m), "success": True, "used_fallback": False,
                    "valhalla_calls": valhalla_calls, "time_s": round(time.time() - start_time, 2),
                    "message": "카카오 API를 사용하여 경로가 단축 후 반환됩니다.", "length_ok": True,
                })
                return best_route, best_meta
        
        # 길이 조건 (±99m) 충족 또는 단축 후 성공
        if abs(final_len - target_m) <= MAX_LENGTH_ERROR_M:
            best_meta.update(
                {
                    "success": True, "used_fallback": False, "km_requested": km_requested, "target_m": target_m,
                    "valhalla_calls": valhalla_calls, "time_s": round(time.time() - start_time, 2),
                    "message": "안정적인 닫힌 루프를 찾았습니다.", "length_ok": True,
                }
            )
            return best_route, best_meta

    # -----------------------------
    # 3. 완전 실패 시: 단순 왕복 시도 (Fallback)
    # -----------------------------
    
    if time.time() - start_time >= GLOBAL_TIMEOUT_S:
         return [start], {"len": 0.0, "err": target_m, "success": False, "used_fallback": False, "valhalla_calls": valhalla_calls, "time_s": round(time.time() - start_time, 2), "message": "경로 생성 요청이 시간 제한(5초)을 초과했습니다."}

    R_fallback = R_MEDIUM * 0.6 
    simple_via = project_point(lat, lng, R_fallback, 0.0)
    out_seg = valhalla_route(start, simple_via); valhalla_calls += 1

    if out_seg and len(out_seg) >= 2:
        back_seg = list(reversed(out_seg))
        
        overlap_index = -1; max_overlap_check = min(len(out_seg), len(back_seg), 10) 
        for k in range(1, max_overlap_check + 1):
            if out_seg[-k] == back_seg[k-1]: overlap_index = k
            else: break
        
        if overlap_index > 0: loop_pts = out_seg[:-overlap_index] + back_seg[overlap_index-1:] 
        else: loop_pts = out_seg + back_seg[1:]

        temp_pts = [loop_pts[0]]; [temp_pts.append(p) for p in loop_pts[1:] if p != temp_pts[-1]]; loop_pts = temp_pts
        
        if loop_pts and loop_pts[0] != start: loop_pts[0] = start
        if loop_pts and loop_pts[-1] != start: loop_pts[-1] = start
            
        fallback_len = polyline_length_m(loop_pts)
        fallback_err = abs(fallback_len - target_m)
        
        # [핵심] Fallback도 ±99m 이내일 때만 허용
        if fallback_err <= MAX_LENGTH_ERROR_M: 
            meta = {
                "len": fallback_len, "err": fallback_err, "roundness": _loop_roundness(loop_pts), 
                "score": fallback_err + (1.0 - _loop_roundness(loop_pts)) * 0.3 * target_m,
                "length_ok": True, "success": False, "used_fallback": True, "km_requested": km_requested, "target_m": target_m,
                "valhalla_calls": valhalla_calls, "time_s": round(time.time() - start_time, 2),
                "message": f"최적 루프를 찾지 못했으나, 길이({fallback_len:.1f}m)가 요청 오차(±{MAX_LENGTH_ERROR_M}m)를 만족하여 반환합니다.",
            }
            return loop_pts, meta

    # 최종 실패
    return [start], {"len": 0.0, "err": target_m, "success": False, "used_fallback": False, "km_requested": km_requested, "target_m": target_m, "valhalla_calls": valhalla_calls, "time_s": round(time.time() - start_time, 2), "message": f"요청 오차(±{MAX_LENGTH_ERROR_M}m)를 만족하는 경로를 찾을 수 없습니다. 거리를 조정해 주세요."}
