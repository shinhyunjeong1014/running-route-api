import math
import random
import logging
import os
from typing import List, Dict, Tuple

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 목표 거리 허용 오차 (m) – 예: 2km 요청 시 1.75~2.25km 정도 허용
TARGET_RANGE_M = 250.0

# Valhalla 라우팅 엔드포인트
VALHALLA_URL = os.getenv("VALHALLA_URL", "http://localhost:8002/route")


# -------------------------------------------------
# 기본 유틸 (turn_algo에서도 사용)
# -------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이의 거리를 미터(m) 단위로 계산 (Haversine)."""
    R = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _cumulative_distances(polyline: List[Dict[str, float]]) -> List[float]:
    """Polyline 각 포인트까지의 누적 거리 리스트를 계산."""
    dists = [0.0]
    for p, q in zip(polyline[:-1], polyline[1:]):
        dists.append(dists[-1] + haversine_m(p["lat"], p["lng"], q["lat"], q["lng"]))
    return dists


# -------------------------------------------------
# 지구 위에서 특정 방향/거리만큼 이동
# -------------------------------------------------
def _move_point(lat: float, lng: float, bearing_deg: float, distance_m: float) -> Dict[str, float]:
    """
    시작점(lat, lng)에서 방위각(bearing_deg) 방향으로 distance_m 만큼 이동한 점을 구한다.
    (구면삼각법; 러닝 코스 생성에는 충분한 정밀도)
    """
    R = 6371000.0
    bearing = math.radians(bearing_deg)

    lat1 = math.radians(lat)
    lon1 = math.radians(lng)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / R)
        + math.cos(lat1) * math.sin(distance_m / R) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(distance_m / R) * math.cos(lat1),
        math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2),
    )

    return {
        "lat": math.degrees(lat2),
        "lng": (math.degrees(lon2) + 540) % 360 - 180,
    }


# -------------------------------------------------
# Valhalla polyline6 디코더
# -------------------------------------------------
def _decode_polyline6(encoded: str) -> List[Dict[str, float]]:
    """Valhalla에서 사용하는 polyline6 문자열을 (lat, lng) 리스트로 변환."""
    coords: List[Dict[str, float]] = []
    index = 0
    lat = 0
    lng = 0
    length = len(encoded)

    while index < length:
        # latitude
        result = 0
        shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        # longitude
        result = 0
        shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coords.append({"lat": lat / 1e6, "lng": lng / 1e6})

    return coords


# -------------------------------------------------
# Valhalla 한 구간 라우팅
# -------------------------------------------------
def _route_segment(start: Dict[str, float], end: Dict[str, float]) -> Tuple[List[Dict[str, float]], float]:
    """
    Valhalla에 start -> end 경로를 요청하고,
    (polyline 좌표 리스트, 경로 길이[m])를 반환.
    """
    payload = {
        "locations": [
            {"lat": start["lat"], "lon": start["lng"]},
            {"lat": end["lat"], "lon": end["lng"]},
        ],
        "costing": "pedestrian",
        "costing_options": {
            "pedestrian": {
                # 러닝에 적합하도록 차도/비포장은 약간 페널티
                "use_hills": 0.3,
                "use_tracks": 0.1,
            }
        },
        "directions_options": {
            "units": "kilometers",
            "narrative": False,
        },
    }

    try:
        resp = requests.post(VALHALLA_URL, json=payload, timeout=5)
    except Exception as e:
        raise RuntimeError(f"Valhalla 접속 오류: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"Valhalla 오류: status={resp.status_code}, body={resp.text[:200]}")

    data = resp.json()
    trip = data.get("trip")
    if not trip or "legs" not in trip:
        raise RuntimeError(f"Valhalla 응답 형식 오류: {data}")

    coords: List[Dict[str, float]] = []
    total_len_m = 0.0

    for leg in trip["legs"]:
        shape = leg.get("shape")
        if not shape:
            continue
        leg_coords = _decode_polyline6(shape)
        if not leg_coords:
            continue

        if not coords:
            coords.extend(leg_coords)
        else:
            coords.extend(leg_coords[1:])  # 앞점 중복 제거

        summary = leg.get("summary", {})
        leg_len_km = float(summary.get("length", 0.0))
        total_len_m += leg_len_km * 1000.0

    if len(coords) < 2 or total_len_m <= 0:
        raise RuntimeError("Valhalla 경로가 비어 있거나 길이 0")

    return coords, total_len_m


# -------------------------------------------------
# 루프 품질 평가 (C-PLUS / Star 공통)
# -------------------------------------------------
def _bearing(a: Dict[str, float], b: Dict[str, float]) -> float:
    """a -> b 방위각 (deg)."""
    lat1 = math.radians(a["lat"])
    lon1 = math.radians(a["lng"])
    lat2 = math.radians(b["lat"])
    lon2 = math.radians(b["lng"])

    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brg = math.degrees(math.atan2(y, x))
    return (brg + 360.0) % 360.0


def _loop_quality(polyline: List[Dict[str, float]], target_m: float) -> Tuple[float, float, int, float]:
    """
    루프 품질을 평가한다.
    반환: (길이오차, backtrack_ratio, uturn_count, score)
    score는 작을수록 좋다.
    """
    if len(polyline) < 4:
        return float("inf"), 1.0, 100, float("inf")

    # 전체 길이 및 오차
    cum = _cumulative_distances(polyline)
    total_len = cum[-1]
    err = abs(total_len - target_m)

    # 세그먼트 중복(왕복) 비율 계산
    seg_counts = {}
    for p, q in zip(polyline[:-1], polyline[1:]):
        key = tuple(
            sorted(
                (
                    (round(p["lat"] * 1e5), round(p["lng"] * 1e5)),
                    (round(q["lat"] * 1e5), round(q["lng"] * 1e5)),
                )
            )
        )
        seg_counts[key] = seg_counts.get(key, 0) + 1

    dup_segments = sum(1 for c in seg_counts.values() if c > 1)
    backtrack_ratio = dup_segments / max(len(seg_counts), 1)

    # U턴/급커브 개수
    uturn_count = 0
    for a, b, c in zip(polyline[:-2], polyline[1:-1], polyline[2:]):
        brg1 = _bearing(a, b)
        brg2 = _bearing(b, c)
        diff = abs((brg2 - brg1 + 540.0) % 360.0 - 180.0)
        if diff > 150.0:
            uturn_count += 1

    # 점수 계산: 거리오차 + 왕복/유턴에 대한 페널티 (조금 더 강하게)
    score = (
        err
        + backtrack_ratio * target_m * 3.0   # 왕복 비율 페널티 강화
        + uturn_count * 120.0                # U턴 페널티 강화
    )
    return err, backtrack_ratio, uturn_count, score


# -------------------------------------------------
# (기존) C-PLUS 삼각 루프 – 폴백용
# -------------------------------------------------
def _build_triangle_loop_c_plus(
    lat: float,
    lng: float,
    km: float,
    max_outer_attempts: int = 6,
    inner_attempts: int = 4,
) -> Tuple[List[Dict[str, float]], float, str]:
    """
    기존 C-PLUS 삼각 루프 알고리즘.
    Star-loop가 실패했을 때 폴백 용도로만 사용한다.
    """
    start = {"lat": lat, "lng": lng}
    target_m = km * 1000.0

    # 초기 반지름 추정
    base_radius = max(300.0, min(target_m * 0.6, target_m / (3.0 * 1.3)))

    best = None  # (polyline, length_m, err, backtrack_ratio, uturn_count, score, tag)

    for outer in range(max_outer_attempts):
        base_bearing = random.uniform(0.0, 360.0)

        for inner in range(inner_attempts):
            radius_factor = 0.8 + 0.4 * random.random()
            radius = base_radius * radius_factor

            jitter = random.uniform(-25.0, 25.0)
            bearing_a = base_bearing + jitter
            bearing_b = base_bearing + 120.0 + jitter

            A = _move_point(lat, lng, bearing_a, radius)
            B = _move_point(lat, lng, bearing_b, radius)

            try:
                seg1, len1 = _route_segment(start, A)
                seg2, len2 = _route_segment(A, B)
                seg3, len3 = _route_segment(B, start)
            except Exception as e:
                logger.info(f"[TriLoop C+] Valhalla 세그먼트 실패 outer={outer}, inner={inner}: {e}")
                continue

            loop_coords: List[Dict[str, float]] = []
            loop_coords.extend(seg1)
            loop_coords.extend(seg2[1:])
            loop_coords.extend(seg3[1:])

            err, backtrack_ratio, uturn_count, score = _loop_quality(loop_coords, target_m)
            total_len = _cumulative_distances(loop_coords)[-1]

            logger.info(
                f"[TriLoop C+] outer={outer}, inner={inner}, radius={radius:.0f}m, "
                f"len={total_len:.1f}m, err={err:.1f}m, br={backtrack_ratio:.2f}, "
                f"uturn={uturn_count}, score={score:.1f}"
            )

            if backtrack_ratio > 0.45:
                continue
            if err > target_m * 0.75:
                continue

            if best is None or score < best[5]:
                best = (
                    loop_coords,
                    total_len,
                    err,
                    backtrack_ratio,
                    uturn_count,
                    score,
                    f"VH_Triangle_C_PLUS r={radius:.0f}m",
                )

            if err <= TARGET_RANGE_M and backtrack_ratio <= 0.25 and uturn_count <= 3:
                return (
                    loop_coords,
                    total_len,
                    f"VH_Triangle_C_PLUS_OK r={radius:.0f}m",
                )

    if best is not None:
        coords, total_len, err, br, uc, score, tag = best
        logger.warning(
            f"[TriLoop C+] 정확히 맞추지 못했지만 최선의 루프 반환: "
            f"len={total_len:.1f}m, target={target_m:.1f}m, err={err:.1f}m, "
            f"br={br:.2f}, uturn={uc}, score={score:.1f}"
        )
        return coords, total_len, tag

    raise RuntimeError("Valhalla 기반 C-PLUS 삼각 루프를 생성하지 못했습니다.")


# -------------------------------------------------
# (새) Star-loop 네트워크 기반 러닝 루프 생성
# -------------------------------------------------
def _build_star_loop_network(
    lat: float,
    lng: float,
    km: float,
    max_outer_attempts: int = 8,
    inner_attempts: int = 6,
) -> Tuple[List[Dict[str, float]], float, str]:
    """
    시작점 기준으로 4~5개의 방사형 포인트를 만들고,
    Start -> P1 -> ... -> Pn -> Start 형태의 러닝 루프를 생성한다.

    - 도로 네트워크를 따라가는 Valhalla 경로를 사용
    - 거리 오차, backtrack 비율, U턴 개수로 품질 평가
    - 점수가 가장 좋은 루프를 선택
    """
    start = {"lat": lat, "lng": lng}
    target_m = km * 1000.0

    # 대략적인 반지름 추정 (대략 원둘레 2πr ~ target_m를 가정)
    # 실제 도로는 직선이 아니므로 조금 보정
    base_radius = target_m / (2.8 * math.pi) * 2.0  # 약간 크게
    base_radius = max(350.0, min(base_radius, 1500.0))

    best = None  # (polyline, total_len, err, br, uc, score, tag)

    for outer in range(max_outer_attempts):
        # 4개 또는 5개 포인트를 사용 (사각형/오각형 형태)
        num_rays = random.choice([4, 5])
        angle_step = 360.0 / num_rays
        base_angle = random.uniform(0.0, 360.0)

        for inner in range(inner_attempts):
            via_points: List[Dict[str, float]] = []

            # 각 ray마다 약간씩 다른 반지름/각도를 부여
            for i in range(num_rays):
                angle = base_angle + i * angle_step + random.uniform(-20.0, 20.0)
                radius_factor = 0.8 + 0.5 * random.random()  # 0.8 ~ 1.3
                radius = base_radius * radius_factor
                via_points.append(_move_point(lat, lng, angle, radius))

            # Valhalla로 Start -> P1 -> ... -> Pn -> Start 경로 생성
            try:
                loop_coords: List[Dict[str, float]] = []
                total_len = 0.0

                current = start
                first = True
                for vp in via_points:
                    seg, seg_len = _route_segment(current, vp)
                    if first:
                        loop_coords.extend(seg)
                        first = False
                    else:
                        loop_coords.extend(seg[1:])
                    total_len += seg_len
                    current = vp

                # 마지막: Start로 복귀
                seg, seg_len = _route_segment(current, start)
                loop_coords.extend(seg[1:])
                total_len += seg_len

            except Exception as e:
                logger.info(f"[StarLoop] Valhalla 세그먼트 실패 outer={outer}, inner={inner}: {e}")
                continue

            err, backtrack_ratio, uturn_count, score = _loop_quality(loop_coords, target_m)
            cum = _cumulative_distances(loop_coords)
            total_len = cum[-1]

            logger.info(
                f"[StarLoop] outer={outer}, inner={inner}, rays={num_rays}, "
                f"base_r={base_radius:.0f}m, len={total_len:.1f}m, "
                f"err={err:.1f}m, br={backtrack_ratio:.2f}, "
                f"uturn={uturn_count}, score={score:.1f}"
            )

            # 품질 필터: 왕복/거리오차가 너무 큰 루프는 탈락
            if backtrack_ratio > 0.30:
                continue
            if err > target_m * 0.5:
                continue
            if total_len < target_m * 0.6:
                continue

            # 최고 루프 갱신
            if best is None or score < best[5]:
                best = (
                    loop_coords,
                    total_len,
                    err,
                    backtrack_ratio,
                    uturn_count,
                    score,
                    f"VH_StarLoop rays={num_rays} r={base_radius:.0f}m",
                )

            # 충분히 좋은 루프면 바로 반환
            if err <= TARGET_RANGE_M and backtrack_ratio <= 0.20 and uturn_count <= 4:
                return (
                    loop_coords,
                    total_len,
                    f"VH_StarLoop_OK rays={num_rays} r={base_radius:.0f}m",
                )

    if best is not None:
        coords, total_len, err, br, uc, score, tag = best
        logger.warning(
            f"[StarLoop] 정확히 맞추지 못했지만 최선의 루프 반환: "
            f"len={total_len:.1f}m, target={target_m:.1f}m, err={err:.1f}m, "
            f"br={br:.2f}, uturn={uc}, score={score:.1f}"
        )
        return coords, total_len, tag

    raise RuntimeError("Valhalla 기반 Star-loop 러닝 루프를 생성하지 못했습니다.")


# -------------------------------------------------
# 외부 인터페이스
# -------------------------------------------------
def generate_route(lat: float, lng: float, km: float):
    """
    FastAPI(app.py)에서 호출하는 외부 인터페이스.
    우선 Star-loop 알고리즘을 사용하고, 실패 시 삼각형 C-PLUS로 폴백한다.
    (polyline_coords, length_m, algorithm_used) 형태로 반환.
    """
    # 1차: Star-loop
    try:
        coords, length_m, algo_tag = _build_star_loop_network(lat, lng, km)
        return coords, length_m, algo_tag
    except Exception as e:
        logger.warning(f"[generate_route] Star-loop 실패, Triangle C-PLUS 폴백: {e}")

    # 2차: Triangle C-PLUS 폴백
    coords, length_m, algo_tag = _build_triangle_loop_c_plus(lat, lng, km)
    return coords, length_m, algo_tag


# 과거 코드 호환용 (혹시 generate_loop_route를 임포트하는 버전이 남아 있다면)
def generate_loop_route(lat: float, lng: float, km: float):
    return generate_route(lat, lng, km)
