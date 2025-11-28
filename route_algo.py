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
# 기본 설정 (유지)
# -----------------------------

VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")
VALHALLA_TIMEOUT = float(os.environ.get("VALHALLA_TIMEOUT", "2.5"))
VALHALLA_MAX_RETRY = int(os.environ.get("VALHALLA_MAX_RETRY", "1"))

KAKAO_API_KEY = "dc3686309f8af498d7c62bed0321ee64"
KAKAO_ROUTE_URL = "https://apis-navi.kakaomobility.com/v1/directions"

RUNNING_SPEED_KMH = 8.0  
GLOBAL_TIMEOUT_S = 10.0 
MAX_TOTAL_CALLS = 30 
MAX_LENGTH_ERROR_M = 99.0

# -----------------------------
# 거리 / 기하 유틸 (유지)
# -----------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
# Valhalla/Kakao API 호출 (유지)
# -----------------------------
# ... (valhalla_route, kakao_walk_route, _decode_polyline 함수는 코드가 길어 생략하며, 이전 버전과 동일하게 유지됩니다.)

# -----------------------------
# 루프 품질 평가 / 안전성 필터 / 단축 재연결 로직
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
    # 이 함수는 현재 로직에서 안전성 필터링 역할은 하지 않습니다.
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


# -----------------------------
# Area Loop 생성 (삼각형 폐쇄 루프 방식)
# -----------------------------

def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
) -> Tuple[List[Tuple[float, float]], Dict]:
    """목표 거리(km) 근처의 '닫힌 러닝 루프'를 생성한다. (길이 초과 시 즉시 카카오 단축 시도)"""
    
    start_time = time.time()
    
    target_m = max(300.0, km * 1000.0) 
    km_requested = km
    start = (lat, lng)

    if time.time() - start_time >= GLOBAL_TIMEOUT_S:
         return [start], {"len": 0.0, "err": target_m, "success": False, "used_fallback": False, "valhalla_calls": 0, "time_s": 0.0, "message": "경로 생성 요청이 시작하자마자 시간 제한(10초)을 초과했습니다."}

    SEGMENT_LEN = target_m / 3.0
    R_ideal = target_m / (2.0 * math.pi)
    
    # [수정] R 제약 완화 (100m까지 낮춰 탐색 공간 최대화)
    R_MIN = max(100.0, min(R_ideal * 0.3, 200.0))
    R_SMALL = max(200.0, min(R_ideal * 0.6, 400.0))
    R_MEDIUM = max(400.0, min(R_ideal * 1.0, 700.0))
    R_LARGE = max(700.0, min(R_ideal * 1.3, 1100.0))
    R_XLARGE = max(1100.0, min(R_ideal * 1.6, 1800.0))
    
    radii = list(sorted(list(set([R_MIN, R_SMALL, R_MEDIUM, R_LARGE, R_XLARGE]))))
    bearings = [0, 90, 180, 270] 

    best_route: List[Tuple[float, float]] = []; best_meta: Dict = {}; best_score = float("inf")
    valhalla_calls = 0

    # 1. 5단계 반경 + 4방위 테스트 (최대 30회 호출)
    for R in radii:
        if valhalla_calls + 3 > MAX_TOTAL_CALLS: break
        if time.time() - start_time >= GLOBAL_TIMEOUT_S: break

        for br in bearings:
            if valhalla_calls + 3 > MAX_TOTAL_CALLS: break
            if time.time() - start_time >= GLOBAL_TIMEOUT_S: break

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
            
            # [핵심 수정] 길이 조건(±99m)에 관계없이, 안전성만 통과하면 best_route로 저장
            # 카카오 단축 로직이 실행될 수 있도록 후보군을 모음
            if _is_path_safe(loop_pts) and final_len > 0 and score < best_score:
                best_score = score
                best_route = loop_pts
                best_meta = local_meta

        if valhalla_calls + 3 > MAX_TOTAL_CALLS: break

    # -----------------------------
    # 2. 결과 정리 (성공 케이스)
    # -----------------------------
    if best_route:
        final_len = polyline_length_m(best_route)
        
        # [핵심] 길이 초과 시 (99m 초과) 카카오 단축 로직 시도
        if abs(final_len - target_m) > MAX_LENGTH_ERROR_M:
            logger.info("[Loop Gen] Path found (len=%d), but error > 99m. Attempting Kakao shrink...", final_len)
            
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
        
        # 길이 조건 (±99m) 충족 (단축 전 또는 단축 후)
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
         return [start], {"len": 0.0, "err": target_m, "success": False, "used_fallback": False, "valhalla_calls": valhalla_calls, "time_s": round(time.time() - start_time, 2), "message": "경로 생성 요청이 시간 제한(10초)을 초과했습니다."}

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
        
        # [핵심] Fallback도 ±99m 이내일 때만 허용 (안전성 필터는 제거됨)
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
