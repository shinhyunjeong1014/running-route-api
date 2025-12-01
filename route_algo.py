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
VALHALLA_MAX_RETRY = int(os.environ.get("VALHALLA_MAX_RETRY", "2")) 

KAKAO_API_KEY = "dc3686309f8af498d7c62bed0321ee64" # 실제 키 사용을 위해 환경 변수 권장
KAKAO_ROUTE_URL = "https://apis-navi.kakaomobility.com/v1/directions"

RUNNING_SPEED_KMH = 8.0  
GLOBAL_TIMEOUT_S = 10.0 
MAX_TOTAL_CALLS = 30 
MAX_LENGTH_ERROR_M = 99.0 # 목표 길이 오차 허용치 (99m 유지)
MAX_BEST_ROUTES_TO_TEST = 5 
MAX_ROUTES_TO_PROCESS = 10 

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

def _get_bounding_box_polygon(points: List[Tuple[float, float]], buffer_deg: float = 0.0002) -> Optional[List[Tuple[float, float]]]:
    """
    [핵심 개선] 경로 중복 회피용 Bounding Box Polygon을 생성합니다.
    (Valhalla의 avoid_polygons를 이용한 '경로 독(Poisoning)' 간접 구현)
    """
    if not points or len(points) < 2: return None
    
    # 경로를 따라 완충 영역을 포함하는 Convex Hull을 만드는 것이 이상적이나, 
    # 여기서는 간단히 경로의 Min/Max Lat/Lon을 기반으로 Bounding Box를 확장합니다.
    min_lat = min(p[0] for p in points); max_lat = max(p[0] for p in points)
    min_lon = min(p[1] for p in points); max_lon = max(p[1] for p in points)
    
    # 너무 짧은 경로는 독을 적용하지 않음 (오차 발생 가능성 줄임)
    if haversine_m(min_lat, min_lon, max_lat, max_lon) < 50: return None
        
    buf = buffer_deg

    return [
        (min_lat - buf, min_lon - buf), (max_lat + buf, min_lon - buf),
        (max_lat + buf, max_lon + buf), (min_lat - buf, max_lon + buf),
        (min_lat - buf, min_lon - buf),
    ]


def valhalla_route(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    avoid_polygons: Optional[List[List[Tuple[float, float]]]] = None,
    is_shrink_attempt: bool = False
) -> List[Tuple[float, float]]:
    lat1, lon1 = p1; lat2, lon2 = p2
    last_error: Optional[Exception] = None
    
    # [핵심] 논문 기반의 안전/도보 선호 Costing Options + U-Turn 페널티 강화
    costing_options = {
        "pedestrian": {
            "avoid_steps": 1.0, 
            "service_penalty": 1000, 
            "use_hills": 0.0,
            "use_ferry": 0.0,
            "track_type_penalty": 0, 
            "private_road_penalty": 100000, # 사유지 회피 극대화
            
            "bicycle_network_preference": 0.5,
            "sidewalk_preference": 1.0, # 보도 선호도 최대화
            "alley_preference": -1.0, # 골목길 회피
            "max_road_class": 0.5, # 차도 회피
            "length_penalty": 0.0, # 긴 경로 탐색 유도
            "turn_penalty": 5000.0 # [개선] 잦은 급회전 (U-Turn) 페널티 강화
        }
    }
    
    for attempt in range(VALHALLA_MAX_RETRY):
        try:
            payload = {
                "locations": [{"lat": lat1, "lon": lon1, "type": "break"}, {"lat": lat2, "lon": lon2, "type": "break"}],
                "costing": "pedestrian",
                "costing_options": costing_options
            }
            if avoid_polygons:
                valhalla_polys = []
                for poly in avoid_polygons:
                    valhalla_polys.append([[lon, lat] for lat, lon in poly])
                
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

def kakao_walk_route(p1: Tuple[float, float], p2: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
    """카카오 길찾기 API (도보)를 호출하여 경로 폴리라인을 반환"""
    if not KAKAO_API_KEY:
        logger.error("[Kakao API] KAKAO_API_KEY not configured.")
        return None
    
    lon1, lat1 = p1[::-1]; lon2, lat2 = p2[::-1]

    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    # priority를 'RECOMMEND'가 아닌 'SHORTEST'로 하여 단축 시도에 더 적합하게 변경
    params = {"origin": f"{lon1},{lat1}", "destination": f"{lon2},{lat2}", "waypoints": "", "priority": "SHORTEST", "car_model": "walk"} 

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
# 루프 품질 평가 / 단축 재연결 로직 (일부 보존 및 통합)
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
    # [핵심] 오차(err) + 원형성(1-roundness) 페널티로 점수 산정
    err = abs(length_m - target_m); roundness = _loop_roundness(points)
    score = err + (1.0 - roundness) * 0.3 * target_m
    length_ok = True 
    return score, {"len": length_m, "err": err, "roundness": roundness, "score": score, "length_ok": length_ok}


def _try_shrink_path_kakao(
    current_route: List[Tuple[float, float]],
    target_m: float,
    start_time: float,
    global_timeout: float,
) -> Optional[List[Tuple[float, float]]]:
    """
    [핵심 개선] 카카오 API를 사용하여 경로를 반복적으로 단축 시도. 
    (단축 시도 횟수를 줄여 타임아웃 방지, 성공률 낮은 Valhalla 재시도 로직 제거)
    """
    
    current_len = polyline_length_m(current_route)
    error_m = current_len - target_m
    
    if abs(error_m) <= MAX_LENGTH_ERROR_M:
        return current_route # 이미 목표 달성

    if error_m < 0:
        return None # 늘려야 하는 경우이므로 단축 불필요

    pts = current_route
    
    # 단축할 구간 선택 (중앙 20% 구간)
    idx_a = max(1, int(len(pts) * 0.40))
    idx_b = min(len(pts) - 2, int(len(pts) * 0.60))
    
    MAX_SHRINK_ATTEMPTS = 3 # 최대 3회 시도
    route_to_shrink = current_route[:]
    
    for attempt in range(MAX_SHRINK_ATTEMPTS):
        if time.time() - start_time >= global_timeout: break
        
        current_len = polyline_length_m(route_to_shrink)
        error_m = current_len - target_m
        
        if abs(error_m) <= MAX_LENGTH_ERROR_M:
            return route_to_shrink # 목표 달성
        if error_m < 0:
            break # 더 이상 단축할 필요 없음

        p_a = route_to_shrink[idx_a]; p_b = route_to_shrink[idx_b]
        reconnect_seg = kakao_walk_route(p_a, p_b) # 카카오 최단 경로 사용
        
        if reconnect_seg and len(reconnect_seg) >= 2:
            seg_len_original = polyline_length_m(route_to_shrink[idx_a : idx_b + 1])
            seg_len_new = polyline_length_m(reconnect_seg)
            reduction = seg_len_original - seg_len_new

            if reduction > 10.0: # 10m 이상 단축 효과가 있다면 경로 교체
                new_route = route_to_shrink[:idx_a] + reconnect_seg + route_to_shrink[idx_b+1:]
                route_to_shrink = new_route[:] # 다음 반복을 위해 경로 업데이트
            else:
                break # 단축 효과 미미

    # 최종 검증 후, 목표 달성했으면 경로 반환, 아니면 None 반환
    if abs(polyline_length_m(route_to_shrink) - target_m) <= MAX_LENGTH_ERROR_M:
        return route_to_shrink
    else:
        return None

def _calculate_overlap_penalty(seg_out: List[Tuple[float, float]], seg_back: List[Tuple[float, float]]) -> float:
    """
    복귀 경로(seg_back)가 나가는 경로(seg_out)와 공간적으로 겹치는 정도를 측정하여 페널티를 부과합니다.
    """
    if not seg_out or not seg_back: return 0.0

    overlap_count = 0
    OVERLAP_THRESHOLD_DEG = 0.0002 # 약 20m 근접성
    
    # out 경로의 점 집합을 만듦 (더 빠른 검색을 위해)
    out_points_set = set(seg_out)
    
    for lat_c, lon_c in seg_back:
        is_close = False
        # 단순히 점 일치만 보는 대신, 일정 근접 거리(0.0002도) 내에 있는지 확인
        for lat_a, lon_a in seg_out:
            if abs(lat_c - lat_a) < OVERLAP_THRESHOLD_DEG and abs(lon_c - lon_a) < OVERLAP_THRESHOLD_DEG:
                is_close = True
                break
        if is_close:
            overlap_count += 1

    seg_back_len = len(seg_back)
    if seg_back_len > 0 and overlap_count / seg_back_len > 0.1: # 10% 이상 겹치면 페널티
        overlap_ratio = overlap_count / seg_back_len
        # 겹침 정도에 따라 페널티를 부과 (최대 1000m 상당)
        return overlap_ratio * 1000.0
        
    return 0.0


# -----------------------------
# Area Loop 생성 (Two-Segment Hybrid)
# -----------------------------

def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
) -> Tuple[List[Tuple[float, float]], Dict]:
    """목표 거리(km) 근처의 '닫힌 러닝 루프'를 생성한다. (경로 독 및 길이 조정 최적화 강화)"""
    
    start_time = time.time()
    
    target_m = max(300.0, km * 1000.0) 
    start = (lat, lng)

    if time.time() - start_time >= GLOBAL_TIMEOUT_S:
         return [start], {"len": 0.0, "err": target_m, "success": False, "used_fallback": False, "valhalla_calls": 0, "time_s": 0.0, "message": "경로 생성 요청이 시작하자마자 시간 제한(10초)을 초과했습니다."}

    # R 제약 완화 및 탐색 반경 설정 (기존 로직 유지)
    R_ideal = target_m / (2.0 * math.pi)
    R_MIN = max(100.0, min(R_ideal * 0.3, 200.0))
    R_SMALL = max(200.0, min(R_ideal * 0.6, 400.0))
    R_MEDIUM = max(400.0, min(R_ideal * 1.0, 700.0))
    R_LARGE = max(700.0, min(R_ideal * 1.3, 1100.0))
    R_XLARGE = max(1100.0, min(R_ideal * 1.6, 1800.0))
    
    # 탐색 반경과 방향 조합
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
            
            # 1.1 Seg A: 출발 → Via A (Out Segment)
            seg_out = valhalla_route(start, via_a); valhalla_calls += 1
            if not seg_out or len(seg_out) < 2: continue
            
            comback_point = seg_out[-1]
            
            # 2. 경로 독(Poisoning) 적용
            avoid_polys = _get_bounding_box_polygon(seg_out)
            avoid_polygons = [avoid_polys] if avoid_polys else None

            # 3. 2차 경로 생성 후보 확보 (Comeback → Start)
            back_segments = []
            
            # 3.1 Valhalla 복귀 경로 (Poisoning 적용)
            if valhalla_calls + 1 <= MAX_TOTAL_CALLS:
                seg_back_v = valhalla_route(comback_point, start, avoid_polygons=avoid_polygons)
                valhalla_calls += 1
                if seg_back_v and len(seg_back_v) >= 2: back_segments.append({"seg": seg_back_v, "source": "Valhalla"})
            
            # 3.2 카카오 복귀 경로 (Poisoning 미적용. 혹시 모를 대체 경로로 확보)
            seg_back_k = kakao_walk_route(comback_point, start)
            if seg_back_k and len(seg_back_k) >= 2: back_segments.append({"seg": seg_back_k, "source": "Kakao"})

            # 4. 최종 루프 구성 및 페널티 적용
            for back_seg_data in back_segments:
                seg_back = back_seg_data["seg"]
                
                # 겹침 페널티 계산
                overlap_penalty = _calculate_overlap_penalty(seg_out, seg_back)
                
                # 중복이 심한 경로는 폐기
                if overlap_penalty > 500.0: continue

                total_route = seg_out + seg_back[1:] 
                # 시작/끝점 보정 및 중복점 제거
                temp_pts = [total_route[0]]; [temp_pts.append(p) for p in total_route[1:] if p != temp_pts[-1]]; total_route = temp_pts
                
                score_base, local_meta = _score_loop(total_route, target_m)
                total_score = score_base + overlap_penalty # 최종 점수에 겹침 페널티 부과
                
                if polyline_length_m(total_route) > 0:
                    candidate_routes.append({
                        "route": total_route, 
                        "valhalla_score": total_score, # 페널티가 포함된 점수
                    })
                    total_routes_checked += 1

        if valhalla_calls + 2 > MAX_TOTAL_CALLS: break

    # -----------------------------
    # 5. 모든 후보 경로 후처리 (카카오 단축 시도)
    # -----------------------------
    
    final_validated_routes = []
    candidate_routes.sort(key=lambda x: x["valhalla_score"])
    
    for i, candidate in enumerate(candidate_routes[:MAX_BEST_ROUTES_TO_TEST]): 
        
        if time.time() - start_time >= GLOBAL_TIMEOUT_S: break

        current_route = candidate['route']
        final_len = polyline_length_m(current_route)
        
        # 5.1 ±99m 초과 경로만 단축 시도
        if abs(final_len - target_m) > MAX_LENGTH_ERROR_M:
            shrunken_route = _try_shrink_path_kakao(
                current_route, target_m, start_time, GLOBAL_TIMEOUT_S
            )

            if shrunken_route:
                # 단축 성공 시, 최종 경로로 추가
                final_score = _score_loop(shrunken_route, target_m)[0]
                final_validated_routes.append({"route": shrunken_route, "score": final_score})
            else:
                # 단축에 실패했더라도, 원본 경로를 최종 후보에 포함 (가장 인접한 경로 확보)
                final_validated_routes.append({"route": current_route, "score": _score_loop(current_route, target_m)[0]})
        else:
            # 이미 목표 달성한 경로
            final_validated_routes.append({"route": current_route, "score": _score_loop(current_route, target_m)[0]})

    # -----------------------------
    # 6. 최종 베스트 경로 선택 (절대 반환 보장)
    # -----------------------------
    
    best_final_route = None
    
    if final_validated_routes:
        final_validated_routes.sort(key=lambda x: x["score"])
        best_final_route = final_validated_routes[0]["route"]
        
    
    # -----------------------------
    # 7. 결과 반환 (성공/최인접 경로 반환 보장)
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
    # 8. 최종 실패 (경로 후보가 0개)
    # -----------------------------
    return [start], {
        "len": 0.0, "err": target_m, "success": False, "used_fallback": False, 
        "valhalla_calls": valhalla_calls, "time_s": round(time.time() - start_time, 2),
        "message": f"탐색 결과, 유효한 경로 후보를 찾을 수 없습니다. (Valhalla 통신 불가 또는 지리적 단절)",
        "routes_checked": total_routes_checked,
    }
