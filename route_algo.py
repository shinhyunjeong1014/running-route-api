import math
import logging
from valhalla import get_route  # 너가 이미 만든 Valhalla wrapper 그대로 사용
from turn_algo import build_turn_by_turn

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

# ----------------------------------------
# 기본 유틸
# ----------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p = math.pi / 180
    lat1, lon1, lat2, lon2 = lat1 * p, lon1 * p, lat2 * p, lon2 * p
    d = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(d))


def dest_point(lat, lon, bearing, dist):
    R = 6371000
    d = dist
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    b = math.radians(bearing)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d / R)
        + math.cos(lat1) * math.sin(d / R) * math.cos(b)
    )
    lon2 = lon1 + math.atan2(
        math.sin(b) * math.sin(d / R) * math.cos(lat1),
        math.cos(d / R) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def poly_length(poly):
    if not poly:
        return 0
    total = 0
    for i in range(len(poly) - 1):
        a = poly[i]
        b = poly[i + 1]
        total += haversine(a["lat"], a["lng"], b["lat"], b["lng"])
    return total


def roundness(center, poly):
    """루프가 '둥근' 형태인지 평가"""
    if not poly:
        return 0
    dists = []
    for pt in poly:
        d = haversine(center[0], center[1], pt["lat"], pt["lng"])
        dists.append(d)
    if not dists:
        return 0
    mean_d = sum(dists) / len(dists)
    var = sum((d - mean_d) ** 2 for d in dists) / len(dists)
    # roundness는 '작을수록' 둥글다 → 0~1 범위로 변환
    return 1 / (1 + math.sqrt(var))


# ----------------------------------------
# A1 Round Loop 알고리즘
# ----------------------------------------

def generate_area_loop_a1(lat, lng, km):
    """A1 = 전국 공통 안정형 둥근 루프 생성기"""
    target = km * 1000
    center = (lat, lng)

    # region 크기(라디얼 확장) — 너무 크면 안정성 떨어짐 → 300~420m 고정
    base_rs = [300, 340, 380, 420]

    # via 개수 = 3~5개 → 둥근 루프
    via_counts = [3, 4, 5]

    candidates = []

    for base_r in base_rs:
        for via_n in via_counts:
            pts = []
            for k in range(via_n):
                bearing = (360 / via_n) * k
                vy_lat, vy_lng = dest_point(lat, lng, bearing, base_r)
                pts.append((vy_lat, vy_lng))

            # via들을 순서대로 연결하고 마지막에 다시 start로 연결 → 루프
            full_poly = []
            fail = False

            for i in range(via_n):
                a = pts[i]
                b = pts[(i + 1) % via_n]  # 루프
                r = get_route(
                    a[0], a[1],
                    b[0], b[1],
                    mode="pedestrian"  # 도보 전용
                )
                if not r or "polyline" not in r or len(r["polyline"]) < 2:
                    fail = True
                    break
                full_poly.extend(r["polyline"])

            if fail:
                continue

            L = poly_length(full_poly)
            err = abs(L - target)
            r_score = roundness(center, full_poly)

            # 후보 저장
            cand = {
                "poly": full_poly,
                "len": L,
                "err": err,
                "round": r_score,
                "base_r": base_r,
                "via": via_n
            }
            candidates.append(cand)

            logger.info(
                f"[AreaLoop-A1] base_r={base_r}, via={via_n}, len={L:.1f}, "
                f"err={err:.1f}, round={r_score:.2f}"
            )

    if not candidates:
        logger.warning("[AreaLoop-A1] No candidates found")
        return {"error": "no route"}

    # ----------------------------------------
    # 최종 후보 선택: 오차 < 100m 우선
    # ----------------------------------------
    tight = [c for c in candidates if c["err"] <= 100]

    if tight:
        # 오차가 비슷하면 둥근 정도(roundness)가 높은 것 우선
        best = sorted(tight, key=lambda c: (c["err"], -c["round"]))[0]
    else:
        # 완벽히 맞는 게 없으면 가장 오차가 작은 후보
        best = sorted(candidates, key=lambda c: (c["err"], -c["round"]))[0]
        logger.warning(
            f"[AreaLoop-A1] 정확히 ±100m 내 경로 없음. 최선 후보 len={best['len']:.1f}, err={best['err']:.1f}"
        )

    # 턴바이턴 생성
    turns = build_turn_by_turn(best["poly"])

    logger.info(
        f"[AreaLoop-A1] FINAL: len={best['len']:.1f}, err={best['err']:.1f}, "
        f"base_r={best['base_r']}, via={best['via']}"
    )

    return {
        "start": {"lat": lat, "lng": lng},
        "distance_m": best["len"],
        "error_m": best["err"],
        "polyline": best["poly"],
        "turn_by_turn": turns
    }
