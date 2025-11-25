import math
import os
import logging
from typing import List, Dict, Tuple

import requests

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

# -----------------------------
# 거리 계산
# -----------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# -----------------------------
# Valhalla API 호출 (도보 전용)
# -----------------------------
VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")
VALHALLA_TIMEOUT = float(os.environ.get("VALHALLA_TIMEOUT", "10"))
VALHALLA_MAX_RETRY = int(os.environ.get("VALHALLA_MAX_RETRY", "2"))


def _decode_polyline(shape: str) -> List[Tuple[float, float]]:
    """
    Valhalla polyline 디코딩 (lat, lng 리스트 반환)
    """
    coords: List[Tuple[float, float]] = []
    lat = 0
    lng = 0
    idx = 0

    while idx < len(shape):
        b = 1
        shift = 0
        result = 0
        # latitude
        while b >= 0x20:
            b = ord(shape[idx]) - 63
            idx += 1
            result |= (b & 0x1F) << shift
            shift += 5
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        # longitude
        b = 1
        shift = 0
        result = 0
        while b >= 0x20:
            b = ord(shape[idx]) - 63
            idx += 1
            result |= (b & 0x1F) << shift
            shift += 5
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coords.append((lat / 1e5, lng / 1e5))

    return coords


def valhalla_route(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """
    Valhalla로 '도보 전용' 경로 요청.
    실패 시 지정된 횟수만큼 재시도 후 빈 리스트 반환.
    """
    body = {
        "locations": [
            {"lat": p1[0], "lon": p1[1]},
            {"lat": p2[0], "lon": p2[1]},
        ],
        "costing": "pedestrian",
        "directions_options": {"units": "meters"},
    }

    last_error = None
    for attempt in range(VALHALLA_MAX_RETRY + 1):
        try:
            r = requests.post(VALHALLA_URL, json=body, timeout=VALHALLA_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            shape = data["trip"]["legs"][0]["shape"]
            coords = _decode_polyline(shape)
            if len(coords) < 2:
                raise ValueError("decoded polyline too short")
            return coords
        except Exception as e:
            last_error = e
            logger.warning(
                "[Valhalla] attempt %d failed for %s -> %s: %s",
                attempt + 1, p1, p2, e
            )

    logger.error(f"[Valhalla] all attempts failed for {p1} -> {p2}: {last_error}")
    return []


# -----------------------------
# 루프 점수 계산 (낮을수록 좋음)
# -----------------------------
def compute_loop_score(length_m, target_m, fail_cnt, roundness):
    """
    루프 품질 점수:
    - 거리 오차
    - 실패(Valhalla 실패/스킵) 횟수
    - 원형도(1에 가까울수록 좋음)
    """
    err = abs(length_m - target_m)

    # 너무 멀어지면 거리 페널티 강화
    if err <= 150:
        dist_penalty = err
    else:
        dist_penalty = 150 + (err - 150) * 2.0

    # roundness가 0.7 이하로 떨어지면 급격히 페널티
    round_penalty = (1 - roundness) * (120 if roundness >= 0.7 else 220)

    return dist_penalty + fail_cnt * 350 + round_penalty


def _estimate_roundness(length_m: float, R: float) -> float:
    """
    실제 둘레 / 이론적 원둘레 ≈ 원형도 (0~1)
    """
    if R <= 0:
        return 0.0
    ideal = 2 * math.pi * R
    if ideal <= 0:
        return 0.0
    r = length_m / ideal
    # 너무 과하게 크거나 작아도 0.2~1.0 사이로 클램프
    return max(0.2, min(0.99, r))


# -----------------------------
# Area-Loop (옵션 A)
#  - 도보 전용 Valhalla 경로 사용
#  - 목표 거리 ±99m 이내 우선 탐색
#  - 실패/기괴 루프는 과감히 "실패"로 돌려보냄
# -----------------------------
def generate_area_loop(lat: float, lng: float, km: float):
    target_m = km * 1000.0
    MAX_ERR = 99.0

    # 검색 범위 (너무 넓게 돌리면 기괴한 루프 확률↑)
    MAX_OUTER = 10
    MAX_INNER = 8

    # 기본 반경 설정 (대략적인 1바퀴 둘레 기준)
    base_r = max(280.0, min(420.0, target_m / 7.0))

    best_score = float("inf")
    best_route: List[Tuple[float, float]] = []
    best_meta: Dict = {}

    # 에러 상관없이 전체 중 최선 후보 (폴백용)
    best_any_score = float("inf")
    best_any_route: List[Tuple[float, float]] = []
    best_any_meta: Dict = {}

    for outer in range(MAX_OUTER):
        R = base_r + outer * 22.0  # 점점 큰 원으로
        for inner in range(MAX_INNER):
            via_cnt = 3 + (inner % 3)  # 3~5개 방사형 포인트

            angle_step = 360.0 / via_cnt
            rad_points: List[Tuple[float, float]] = []
            for i in range(via_cnt):
                ang = math.radians(i * angle_step)

                rad_lat = lat + (R / 111111.0) * math.cos(ang)
                rad_lng = lng + (R / (111111.0 * math.cos(math.radians(lat)))) * math.sin(ang)
                rad_points.append((rad_lat, rad_lng))

            full: List[Tuple[float, float]] = [(lat, lng)]
            fail_cnt = 0

            # 중심 → 방사형 포인트 순차 방문
            for i in range(via_cnt):
                seg = valhalla_route(full[-1], rad_points[i])
                if not seg:
                    fail_cnt += 1
                    continue
                # 중복 포인트 제거
                full.extend(seg[1:])

            # 마지막: 다시 출발점으로 복귀
            back_seg = valhalla_route(full[-1], (lat, lng))
            if back_seg:
                full.extend(back_seg[1:])
            else:
                fail_cnt += 1

            # 유효하지 않은 루트는 스킵
            if len(full) < 4:
                continue

            # 전체 길이 계산
            length_m = 0.0
            for i in range(1, len(full)):
                length_m += haversine_m(
                    full[i - 1][0], full[i - 1][1],
                    full[i][0], full[i][1],
                )

            err = abs(length_m - target_m)

            # 너무 짧거나 너무 긴 루프는 "기괴" 가능성이 높으므로 과감히 버림
            # (0.6~1.6배 범위 안쪽만 정상 후보로 취급)
            if not (0.6 * target_m <= length_m <= 1.6 * target_m):
                logger.info(
                    "[AreaLoop] skip abnormal length outer=%d, inner=%d, len=%.1fm, target=%.1fm",
                    outer, inner, length_m, target_m,
                )
                # 그래도 fallback 후보에는 넣어볼 수 있음
                roundness_rough = _estimate_roundness(length_m, R)
                score_any = compute_loop_score(length_m, target_m, fail_cnt, roundness_rough)
                if score_any < best_any_score:
                    best_any_score = score_any
                    best_any_route = full[:]
                    best_any_meta = {
                        "outer": outer,
                        "inner": inner,
                        "base_r": R,
                        "via": via_cnt,
                        "len": length_m,
                        "err": err,
                        "fail": fail_cnt,
                        "uturn": fail_cnt,  # 하위 호환용
                        "round": roundness_rough,
                        "score": score_any,
                        "abnormal_length": True,
                    }
                continue

            # 원형도 추정
            roundness = _estimate_roundness(length_m, R)
            score = compute_loop_score(length_m, target_m, fail_cnt, roundness)

            meta = {
                "outer": outer,
                "inner": inner,
                "base_r": R,
                "via": via_cnt,
                "len": length_m,
                "err": err,
                "fail": fail_cnt,
                "uturn": fail_cnt,  # 기존 키 이름 유지
                "round": roundness,
                "score": score,
                "abnormal_length": False,
            }

            # 1) 목표거리 ±MAX_ERR 이내 후보 중에서 점수 최소
            if err <= MAX_ERR and score < best_score:
                best_score = score
                best_route = full[:]
                best_meta = meta
                logger.info(f"[AreaLoop] MATCH outer={outer}, inner={inner}, {meta}")
            else:
                logger.info(
                    "[AreaLoop] outer=%d, inner=%d, len=%.1fm, err=%.1fm, "
                    "fail=%d, round=%.2f, score=%.1f",
                    outer, inner, length_m, err, fail_cnt, roundness, score
                )

            # 2) 에러 상관없이 전체 중 최선 후보 (폴백용)
            if score < best_any_score:
                best_any_score = score
                best_any_route = full[:]
                best_any_meta = meta

    # -----------------------------
    # 최종 선택 및 안전 실패 처리
    # -----------------------------
    # 1) MAX_ERR 이내 최선 루프
    if best_route:
        best_meta["success"] = True
        best_meta["used_fallback"] = False
        return best_route, best_meta

    # 2) 에러 범위는 벗어나지만 그나마 괜찮은 루프 (폴백)
    #    너무 멀거나/짧은 경우는 어차피 위에서 필터링 했으므로
    if best_any_route:
        # 그래도 target 대비 0.5~1.8배는 넘지 않게 2차 필터
        length = best_any_meta.get("len", 0.0)
        if 0.5 * target_m <= length <= 1.8 * target_m:
            logger.warning(
                "[AreaLoop] MAX_ERR 내 루프를 찾지 못해 최선 루프 반환: "
                "len=%.1fm, target=%.1fm, err=%.1fm, score=%.1f",
                best_any_meta.get("len", 0.0),
                target_m,
                best_any_meta.get("err", 0.0),
                best_any_meta.get("score", 0.0),
            )
            best_any_meta["success"] = False
            best_any_meta["used_fallback"] = True
            return best_any_route, best_any_meta

    # 3) Valhalla가 전부 실패한 극단적인 경우:
    #    여기서는 "경로 생성 실패"라고 명확히 알리기 위해
    #    시작점만 반환하고 success=False로 표시
    logger.error("[AreaLoop] Valhalla 실패로 유효한 루프를 만들지 못했습니다.")
    return [(lat, lng)], {
        "len": 0.0,
        "err": target_m,
        "success": False,
        "used_fallback": False,
        "message": "Valhalla 경로 생성 실패",
    }
