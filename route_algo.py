import math
import os
import time
import logging
from typing import List, Dict, Tuple, Optional

import requests

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 기본 설정 (강화)
# -----------------------------

VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")
# 개별 요청 타임아웃: 3초 -> 2.5초 단축
VALHALLA_TIMEOUT = float(os.environ.get("VALHALLA_TIMEOUT", "2.5"))
# 재시도 횟수: 2회 -> 1회 (Valhalla 호출 수 제한을 위해)
VALHALLA_MAX_RETRY = int(os.environ.get("VALHALLA_MAX_RETRY", "1"))

# 러닝 속도(분당 km) – 요약 정보에만 사용
RUNNING_SPEED_KMH = 8.0  # 8km/h 기준

# 루프 생성 최대 Valhalla 호출 수 (안전을 위해 최대 16회 이내)
MAX_TOTAL_CALLS = 16

# -----------------------------
# 거리 / 기하 유틸
# -----------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 사이의 대략적인 거리(m)."""
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
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
    """p1 → p2 방위각 (deg, 0=북, 시계방향)."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def project_point(
    lat: float,
    lon: float,
    distance_m: float,
    bearing_deg_: float,
) -> Tuple[float, float]:
    """시작점에서 특정 거리/방위각만큼 이동한 위경도 (단순 구면 좌표)."""
    R = 6371000.0
    br = math.radians(bearing_deg_)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(distance_m / R)
        + math.cos(phi1) * math.sin(distance_m / R) * math.cos(br)
    )
    lam2 = lam1 + math.atan2(
        math.sin(br) * math.sin(distance_m / R) * math.cos(phi1),
        math.cos(distance_m / R) - math.sin(phi1) * math.sin(phi2),
    )

    return (math.degrees(phi2), (math.degrees(lam2) + 540.0) % 360.0 - 180.0)


# -----------------------------
# Valhalla polyline 디코딩 (1e6 정밀도)
# -----------------------------

def _decode_polyline(shape: str) -> List[Tuple[float, float]]:
    """Valhalla의 $10^6$ 정밀도 인코딩 폴리라인 디코딩 및 좌표 검증."""
    coords: List[Tuple[float, float]] = []
    lat = 0
    lng = 0
    idx = 0
    precision = 1e6 # Valhalla 기본 정밀도 (Google Maps는 1e5)

    try:
        while idx < len(shape):
            # lat
            shift = 0
            result = 0
            while True:
                b = ord(shape[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat

            # lng
            shift = 0
            result = 0
            while True:
                b = ord(shape[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            dlng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += dlng

            # 정밀도 적용
            current_lat = lat / precision
            current_lng = lng / precision

            # 좌표 범위 Sanity Check (필수 요구사항)
            if not (-90.0 <= current_lat <= 90.0 and -180.0 <= current_lng <= 180.0):
                logger.error(f"[Valhalla Decode] Sanity check failed: ({current_lat}, {current_lng})")
                return []

            coords.append((current_lat, current_lng))

    except IndexError:
        logger.error("[Valhalla Decode] Unexpected end of polyline string.")
        return []

    return coords


# -----------------------------
# Valhalla API 호출 (도보 전용)
# -----------------------------

def valhalla_route(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Valhalla로 도보 경로를 요청하고 polyline 좌표 리스트를 반환."""
    lat1, lon1 = p1
    lat2, lon2 = p2

    last_error: Optional[Exception] = None

    for attempt in range(VALHALLA_MAX_RETRY):
        try:
            payload = {
                "locations": [
                    {"lat": lat1, "lon": lon1, "type": "break"},
                    {"lat": lat2, "lon": lon2, "type": "break"},
                ],
                "costing": "pedestrian",
            }
            resp = requests.post(
                VALHALLA_URL,
                json=payload,
                timeout=VALHALLA_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            # Valhalla는 항상 shape을 반환한다고 가정
            shape = data["trip"]["legs"][0]["shape"]
            coords = _decode_polyline(shape)
            if len(coords) < 2:
                # 디코딩 실패 또는 너무 짧은 경로
                raise ValueError("decoded polyline too short or invalid")
            return coords
        except Exception as e:
            last_error = e
            logger.warning(
                "[Valhalla] attempt %d failed for %s -> %s: %s",
                attempt + 1,
                p1,
                p2,
                e,
            )

    logger.error("[Valhalla] all attempts failed for %s -> %s: %s", p1, p2, last_error)
    return []


# -----------------------------
# 루프 품질 평가
# -----------------------------

def _loop_roundness(points: List[Tuple[float, float]]) -> float:
    """루프의 '원형도'를 0~1 사이로 대략 계산."""
    if len(points) < 4:
        return 0.0

    xs = [p[1] for p in points]
    ys = [p[0] for p in points]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    # 1. 중심점까지 거리
    dists = [haversine_m(cy, cx, lat, lon) for lat, lon in points]
    if not dists:
        return 0.0

    mean_r = sum(dists) / len(dists)
    if mean_r <= 0:
        return 0.0

    # 2. 분산 (표준편차) 계산
    var = sum((d - mean_r) ** 2 for d in dists) / len(dists)
    
    # 3. 분산이 작을수록 roundness ↑ (경험적 스케일링)
    # var / (mean_r * mean_r + 1e-6) : 변동 계수 제곱
    score = 1.0 / (1.0 + var / (mean_r * mean_r + 1e-6))
    return max(0.0, min(1.0, score))


def _score_loop(
    points: List[Tuple[float, float]],
    target_m: float,
    min_turns: Optional[int] = None, # fallback 검증용
) -> Tuple[float, Dict]:
    length_m = polyline_length_m(points)
    
    if length_m <= 0.0:
        return float("inf"), {
            "len": 0.0,
            "err": target_m,
            "roundness": 0.0,
            "score": float("inf"),
        }

    err = abs(length_m - target_m)
    roundness = _loop_roundness(points)

    # 오차 1m당 1점 + roundness 페널티
    # roundness 페널티: 최대 0.3 * target_m (루프가 비정형일수록 점수↑)
    score = err + (1.0 - roundness) * 0.3 * target_m

    meta = {
        "len": length_m,
        "err": err,
        "roundness": roundness,
        "score": score,
    }
    
    # 추가 검증: 길이 적합성
    # 일반 루프 생성에서는 이 함수가 score 계산에 사용됨
    # fallback에서는 strict한 길이 검증 조건으로 활용됨
    meta["length_ok"] = (abs(length_m - target_m) <= 300.0) # 엄격한 ± 300m 검증

    return score, meta


# -----------------------------
# Area Loop 생성
# -----------------------------

def generate_area_loop(
    lat: float,
    lng: float,
    km: float,
):
    """목표 거리(km) 근처의 '짧은 러닝 루프'를 생성한다. (안정화 버전)"""
    
    start_time = time.time()
    
    # 목표 거리 (최소 300m 확보)
    target_m = max(300.0, km * 1000.0) 
    km_requested = km

    # 이상적인 원의 반지름 R (L = 2πR)
    ideal_R = target_m / (2.0 * math.pi)
    
    # 3단계 가변 반경 테스트 (도심/공원/골목 적응)
    R_SMALL = max(150.0, min(ideal_R * 0.8, 300.0))
    R_MEDIUM = max(300.0, min(ideal_R, 600.0))
    R_LARGE = max(450.0, min(ideal_R * 1.2, 1000.0))
    
    radii = list(sorted(list(set([R_SMALL, R_MEDIUM, R_LARGE]))))

    # 후보 방위각 (8방위)
    bearings = [0, 45, 90, 135, 180, 225, 270, 315]

    best_route: List[Tuple[float, float]] = []
    best_meta: Dict = {}
    best_score = float("inf")

    valhalla_calls = 0
    start = (lat, lng)

    # 1. 3단계 반경 + 8방위 테스트 (최대 3*8*2 = 48회 호출 가능성 -> MAX_TOTAL_CALLS 로 제한)
    for R in radii:
        for br in bearings:
            if valhalla_calls + 2 > MAX_TOTAL_CALLS:
                logger.warning(f"[Loop Gen] Max Valhalla calls limit ({MAX_TOTAL_CALLS}) reached.")
                break # 다음 R 단계/루프 전체 종료

            via = project_point(lat, lng, R, br)

            # 1) 출발 → via
            out_seg = valhalla_route(start, via)
            valhalla_calls += 1

            if not out_seg or len(out_seg) < 2:
                continue

            # 2) via → 출발
            back_seg = valhalla_route(out_seg[-1], start)
            valhalla_calls += 1
            
            if not back_seg or len(back_seg) < 2:
                continue

            # 왕복 루프 polyline 구성 (접점 중복 제거)
            # out_seg: p0 -> p1 -> ... -> p_via
            # back_seg: p_via -> p_n -> ... -> p_start
            loop_pts: List[Tuple[float, float]] = out_seg + back_seg[1:]

            score, local_meta = _score_loop(loop_pts, target_m)
            
            if score < best_score and local_meta["length_ok"]: # 길이 sanity-check 통과한 것만
                best_score = score
                best_route = loop_pts
                best_meta = local_meta

        if valhalla_calls + 2 > MAX_TOTAL_CALLS:
            break

    # -----------------------------
    # 2. 결과 정리 (성공 케이스)
    # -----------------------------
    if best_route:
        length_m = best_meta.get("len", polyline_length_m(best_route))
        
        # NOTE: turn_algo.py 에서 simplify를 수행하므로, 여기서 추가 smooth/simplify는 생략
        
        best_meta.update(
            {
                "success": True,
                "used_fallback": False,
                "km_requested": km_requested,
                "target_m": target_m,
                "valhalla_calls": valhalla_calls,
                "time_s": round(time.time() - start_time, 2),
            }
        )
        return best_route, best_meta

    # -----------------------------
    # 3. 완전 실패 시: 가장 단순한 out-and-back 시도 (엄격 검증)
    # -----------------------------
    
    # 북쪽 0도 방향으로 중간 반경 R_MEDIUM * 0.6 만큼 이동 시도
    R_fallback = R_MEDIUM * 0.6 
    simple_via = project_point(lat, lng, R_fallback, 0.0)
    
    out_seg = valhalla_route(start, simple_via)
    valhalla_calls += 1

    if out_seg and len(out_seg) >= 2:
        # back_seg는 out_seg를 역순으로 사용 (Valhalla 호출 1회 절약)
        # 왕복 경로 구성: 중복 지점 제거
        back_seg = list(reversed(out_seg))
        loop_pts = out_seg + back_seg[1:]

        _, meta = _score_loop(loop_pts, target_m)
        
        # 엄격한 길이 검증 (필수 요구 사항: target_m ± 300m)
        if meta["length_ok"]:
             # NOTE: Fallback 경로는 "turn-by-turn 검사에서 최소 1개 이상의 의미 있는 변화가 있음" 
             # 요구사항을 만족하기 위해 turn_algo.py에서 build_turn_by_turn()을 호출하여 검증해야 하지만, 
             # 이 함수 내에서 외부 모듈을 호출할 수 없으므로, app.py에서 최종 검증 후 status=error를 반환하도록 합니다.
            
            meta.update(
                {
                    "success": False,
                    "used_fallback": True,
                    "km_requested": km_requested,
                    "target_m": target_m,
                    "valhalla_calls": valhalla_calls,
                    "time_s": round(time.time() - start_time, 2),
                    "message": "안전한 루프를 찾지 못해 단순 왕복 경로를 사용했습니다.",
                }
            )
            return loop_pts, meta

    # Valhalla 자체가 완전히 실패했거나, Fallback 경로가 부적합한 경우
    return [start], {
        "len": 0.0,
        "err": target_m,
        "success": False,
        "used_fallback": False,
        "km_requested": km_requested,
        "target_m": target_m,
        "valhalla_calls": valhalla_calls,
        "time_s": round(time.time() - start_time, 2),
        "message": "Valhalla 경로 생성 실패 또는 부적합한 Fallback 경로",
    }
