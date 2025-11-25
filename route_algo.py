import math
import random
import logging
import os
from typing import List, Dict, Tuple

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 러닝앱 x.xkm 표시에 맞추기 위한 최대 허용 거리 오차 (미터)
MAX_ERR = 99.0

# Valhalla 라우팅 엔드포인트
VALHALLA_URL = os.getenv("VALHALLA_URL", "http://localhost:8002/route")


# -------------------------------------------------
# 기본 유틸
# -------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이 거리 (m, Haversine)."""
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
    """Polyline 각 포인트까지 누적 거리 리스트."""
    dists = [0.0]
    for p, q in zip(polyline[:-1], polyline[1:]):
        dists.append(dists[-1] + haversine_m(p["lat"], p["lng"], q["lat"], q["lng"]))
    return dists


def _move_point(lat: float, lng: float, bearing_deg: float, distance_m: float) -> Dict[str, float]:
    """
    시작점(lat, lng)에서 bearing_deg 방향으로 distance_m만큼 이동한 새 좌표를 계산.
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

    return {"lat": math.degrees(lat2), "lng": (math.degrees(lon2) + 540) % 360 - 180}


# -------------------------------------------------
# Valhalla polyline6 디코더
# -------------------------------------------------
def _decode_polyline6(encoded: str) -> List[Dict[str, float]]:
    """Valhalla polyline6 문자열을 (lat,lng) 리스트로 변환."""
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
    (polyline 좌표 리스트, 경로 길이[m]) 반환.
    """
    payload = {
        "locations": [
            {"lat": start["lat"], "lon": start["lng"]},
            {"lat": end["lat"], "lon": end["lng"]},
        ],
        "costing": "pedestrian",
        "costing_options": {
            "pedestrian": {
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
            coords.extend(leg_coords[1:])  # 앞 점 중복 제거

        summary = leg.get("summary", {})
        leg_len_km = float(summary.get("length", 0.0))
        total_len_m += leg_len_km * 1000.0

    if len(coords) < 2 or total_len_m <= 0:
        raise RuntimeError("Valhalla 경로가 비어 있거나 길이 0")

    return coords, total_len_m


# -------------------------------------------------
# 루프 품질 평가
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
    루프 품질 평가.
    반환: (길이오차, backtrack_ratio, uturn_count, score)  – score는 작을수록 좋음.
    """
    if len(polyline) < 4:
        return float("inf"), 1.0, 100, float("inf")

    # 거리/오차
    cum = _cumulative_distances(polyline)
    total_len = cum[-1]
    err = abs(total_len - target_m)

    # 세그먼트 중복(왕복) 비율
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

    # 점수: 거리오차 + 왕복/유턴 페널티
    score = err + backtrack_ratio * target_m * 3.0 + uturn_count * 150.0
    return err, backtrack_ratio, uturn_count, score


def _roundness(via_points: List[Dict[str, float]], center: Dict[str, float]) -> float:
    """
    via 포인트들이 얼마나 '원형'에 가까운지 측정.
    (max_r - min_r) / 평균반경  값. 0에 가까울수록 동심원에 가까움.
    """
    if not via_points:
        return 1.0

    radii = [
        haversine_m(center["lat"], center["lng"], p["lat"], p["lng"])
        for p in via_points
    ]
    avg_r = sum(radii) / len(radii)
    if avg_r <= 1e-6:
        return 1.0
    return (max(radii) - min(radii)) / avg_r


# -------------------------------------------------
# 폴백용 C-PLUS 삼각 루프
# -------------------------------------------------
def _build_triangle_loop_c_plus(
    lat: float,
    lng: float,
    km: float,
    max_outer_attempts: int = 6,
    inner_attempts: int = 4,
):
    """
    C-PLUS 삼각 루프. Area-loop 실패 시 폴백용으로만 사용.
    """
    start = {"lat": lat, "lng": lng}
    target_m = km * 1000.0

    base_radius = max(300.0, min(target_m * 0.6, target_m / (3.0 * 1.3)))

    best = None  # (coords, total_len, err, br, uc, score, tag)

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
                seg1, _ = _route_segment(start, A)
                seg2, _ = _route_segment(A, B)
                seg3, _ = _route_segment(B, start)
            except Exception as e:
                logger.info(f"[TriLoop C+] Valhalla 세그먼트 실패 outer={outer}, inner={inner}: {e}")
                continue

            loop_coords: List[Dict[str, float]] = []
            loop_coords.extend(seg1)
            loop_coords.extend(seg2[1:])
            loop_coords.extend(seg3[1:])

            err, br, uc, score = _loop_quality(loop_coords, target_m)
            total_len = _cumulative_distances(loop_coords)[-1]

            logger.info(
                f"[TriLoop C+] outer={outer}, inner={inner}, radius={radius:.0f}m, "
                f"len={total_len:.1f}m, err={err:.1f}m, br={br:.2f}, "
                f"uturn={uc}, score={score:.1f}"
            )

            if br > 0.5:
                continue

            # 베스트 갱신
            if best is None or score < best[5]:
                best = (
                    loop_coords,
                    total_len,
                    err,
                    br,
                    uc,
                    score,
                    f"VH_Triangle_C_PLUS r={radius:.0f}m",
                )

            # 오차/품질 기준 만족하면 조기 종료
            if err <= MAX_ERR and br <= 0.25 and uc <= 3:
                return loop_coords, total_len, f"VH_Triangle_C_PLUS_OK r={radius:.0f}m"

    # MAX_ERR 내 루프는 못 찾았지만, 그중 최선 반환
    if best is not None:
        coords, total_len, err, br, uc, score, tag = best
        logger.warning(
            f"[TriLoop C+] MAX_ERR 내 루프를 찾지 못해 최선 루프 반환: "
            f"len={total_len:.1f}m, target={target_m:.1f}m, err={err:.1f}m, "
            f"br={br:.2f}, uturn={uc}, score={score:.1f}"
        )
        return coords, total_len, tag

    raise RuntimeError("Valhalla 기반 C-PLUS 삼각 루프를 생성하지 못했습니다.")


# -------------------------------------------------
# 옵션 A: 라운드형 Area-Loop
# -------------------------------------------------
def _build_area_loop(
    lat: float,
    lng: float,
    km: float,
    max_outer_attempts: int = 10,
    inner_attempts: int = 8,
):
    """
    옵션 A: 둥근(라운드형) 러닝 루프 생성.

    - via point를 거의 같은 반경에서 균등 각도로 배치해서
      최대한 '원형'에 가깝게 만든다.
    - backtrack, 유턴, 비원형(radiial 편차) 루프를 강하게 제거.
    """
    start = {"lat": lat, "lng": lng}
    target_m = km * 1000.0

    # 2πr ≈ target_m → 이상적인 반지름
    ideal_r = max(200.0, target_m / (2.0 * math.pi))

    best = None  # (coords, total_len, err, br, uc, roundness, score, tag)

    for outer in range(max_outer_attempts):
        # 옵션 A: 이상 반지름의 1.05~1.35 배에서 선택 (조금 크게)
        r_factor_outer = 1.05 + 0.30 * random.random()
        base_radius = ideal_r * r_factor_outer
        base_radius = max(250.0, min(base_radius, 2000.0))

        for inner in range(inner_attempts):
            # 거리별 via 개수
            if km <= 2.0:
                num_via = 4
            elif km <= 4.0:
                num_via = random.choice([4, 5])
            else:
                num_via = random.choice([5, 6])

            # 각도 균등 분배 + 약간의 지터 (라운드형 유지)
            angle_step = 360.0 / num_via
            base_angle = random.uniform(0.0, 360.0)

            via_points: List[Dict[str, float]] = []
            for i in range(num_via):
                angle = base_angle + i * angle_step
                jitter = random.uniform(-18.0, 18.0)  # 섹터 내 작은 변화
                angle_j = angle + jitter

                # 반경은 base_radius에서 ±10% 이내로만 변화 → 원형 유지
                r_factor = random.uniform(0.9, 1.1)
                r = base_radius * r_factor

                p = _move_point(lat, lng, angle_j, r)
                via_points.append(p)

            # 라운드형 점수 (옵션 A 핵심): radial 편차가 너무 크면 버림
            rnd = _roundness(via_points, start)
            if rnd > 0.35:
                # 너무 찌그러진 다각형(팔이 쭉 뻗은 모양)은 패스
                continue

            # Start -> via1 -> ... -> viaN -> Start
            try:
                loop_coords: List[Dict[str, float]] = []
                current = start
                first = True

                for vp in via_points:
                    seg, _seg_len = _route_segment(current, vp)
                    if first:
                        loop_coords.extend(seg)
                        first = False
                    else:
                        loop_coords.extend(seg[1:])
                    current = vp

                seg, _seg_len = _route_segment(current, start)
                loop_coords.extend(seg[1:])

            except Exception as e:
                logger.info(f"[AreaLoop] Valhalla 세그먼트 실패 outer={outer}, inner={inner}: {e}")
                continue

            err, br, uc, score_base = _loop_quality(loop_coords, target_m)
            total_len = _cumulative_distances(loop_coords)[-1]

            # roundness도 점수에 포함 (작을수록 좋다)
            score = score_base + rnd * target_m * 2.0

            logger.info(
                f"[AreaLoop] outer={outer}, inner={inner}, via={num_via}, "
                f"base_r={base_radius:.0f}m, len={total_len:.1f}m, "
                f"err={err:.1f}m, br={br:.2f}, uturn={uc}, "
                f"round={rnd:.2f}, score={score:.1f}"
            )

            # 너무 짧거나 백트래킹 심한 후보는 바로 제거
            if total_len < target_m * 0.6:
                continue
            if br > 0.35:
                continue
            if uc > 4:
                continue

            # 베스트 갱신 로직
            if best is None:
                best = (
                    loop_coords,
                    total_len,
                    err,
                    br,
                    uc,
                    rnd,
                    score,
                    f"VH_AreaLoop via={num_via} r={base_radius:.0f}m",
                )
            else:
                _, _, best_err, best_br, best_uc, best_rnd, best_score, _ = best

                # 1) MAX_ERR 이내 루프를 우선
                if err <= MAX_ERR and best_err > MAX_ERR:
                    better = True
                elif err <= MAX_ERR and best_err <= MAX_ERR:
                    # 둘 다 MAX_ERR 이내면 score + roundness + br, uc를 종합
                    better = score < best_score
                else:
                    # 둘 다 MAX_ERR 밖이면 score만 비교
                    better = score < best_score

                if better:
                    best = (
                        loop_coords,
                        total_len,
                        err,
                        br,
                        uc,
                        rnd,
                        score,
                        f"VH_AreaLoop via={num_via} r={base_radius:.0f}m",
                    )

            # 조기 종료 조건 (러닝앱 기준 + 라운드형 품질)
            if (
                err <= MAX_ERR
                and br <= 0.25
                and uc <= 3
                and rnd <= 0.25
            ):
                return loop_coords, total_len, f"VH_AreaLoop_OK via={num_via} r={base_radius:.0f}m"

    # MAX_ERR 이내 못 찾았어도, 그중 최선 루프 반환
    if best is not None:
        coords, total_len, err, br, uc, rnd, score, tag = best
        logger.warning(
            f"[AreaLoop] MAX_ERR 내 루프를 찾지 못해 최선 루프 반환: "
            f"len={total_len:.1f}m, target={target_m:.1f}m, err={err:.1f}m, "
            f"br={br:.2f}, uturn={uc}, round={rnd:.2f}, score={score:.1f}"
        )
        return coords, total_len, tag

    raise RuntimeError("Valhalla 기반 Area-loop 러닝 루프를 생성하지 못했습니다.")


# -------------------------------------------------
# 외부 인터페이스
# -------------------------------------------------
def generate_route(lat: float, lng: float, km: float):
    """
    FastAPI(app.py)에서 호출하는 메인 함수.
    1순위: 옵션 A 라운드형 Area-loop
    실패 시: Triangle C-PLUS 폴백
    """
    try:
        coords, length_m, algo_tag = _build_area_loop(lat, lng, km)
        return coords, length_m, algo_tag
    except Exception as e:
        logger.warning(f"[generate_route] Area-loop 실패, Triangle C-PLUS 폴백: {e}")

    coords, length_m, algo_tag = _build_triangle_loop_c_plus(lat, lng, km)
    return coords, length_m, algo_tag


# 과거 이름 호환용
def generate_loop_route(lat: float, lng: float, km: float):
    return generate_route(lat, lng, km)
