from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
import random
from typing import List, Dict, Tuple

from route_algo import generate_area_loop, polyline_length_m
from turn_algo import build_turn_by_turn

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _format_polyline_for_frontend(
    polyline: List[Tuple[float, float]],
) -> List[Dict[str, float]]:
    """ [[lat,lng], ...] → [{"lat":.., "lng":..}, ...] """
    return [{"lat": lat, "lng": lng} for lat, lng in polyline]


@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 지점 위도"),
    lng: float = Query(..., description="시작 지점 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
):
    """
    공격적 재탐색(Aggressive Retry) 버전
    - 매 시도마다 랜덤 시드 변경
    - 탐색 실패 시 총 12회까지 재탐색
    - 성공 기준은 route_algo의 success=True 판정
    """
    start_point_dict = {"lat": lat, "lng": lng}

    MAX_RETRY = 12          # 🔥 재탐색 횟수 증가
    RETRY_DELAY = 0.05      # 🔥 더 빠르게 재탐색

    best_attempt_poly = None
    best_attempt_meta = None

    # ==========================================
    # 🔄 공격적 재탐색 루프
    # ==========================================
    for attempt in range(1, MAX_RETRY + 2):  # 첫 시도 + 재시도들
        # 🔥 attempt 기반 seed 변화 → 완전 다른 경로 탐색
        random.seed(time.time() + attempt * 17)

        polyline_tuples, meta = generate_area_loop(lat, lng, km)

        is_valid_route = (
            polyline_tuples
            and len(polyline_tuples) >= 2
            and polyline_length_m(polyline_tuples) > 0
        )

        # fallback용 기록(가장 인접한 valid route 저장)
        if is_valid_route:
            if best_attempt_poly is None:
                best_attempt_poly = polyline_tuples
                best_attempt_meta = meta
            else:
                # 🔥 fallback 후보 품질 개선: 요청거리와 더 가까운 경로로 갱신
                prev_err = abs(best_attempt_meta["len"] - best_attempt_meta["target_m"])
                new_err = abs(meta["len"] - meta["target_m"])
                if new_err < prev_err:
                    best_attempt_poly = polyline_tuples
                    best_attempt_meta = meta

            # 성공 조건 충족 → 즉시 반환
            if meta.get("success", False):
                turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)

                meta["message"] = (
                    "요청 거리보다 0~99m 이내로 긴 정밀 경로가 도출되었습니다."
                )

                formatted_poly = _format_polyline_for_frontend(polyline_tuples)

                return {
                    "status": "ok",
                    "start": start_point_dict,
                    "polyline": formatted_poly,
                    "turns": turns,
                    "summary": summary,
                    "meta": meta,
                }

        # 실패 → 재탐색
        if attempt <= MAX_RETRY:
            time.sleep(RETRY_DELAY)
            continue
        else:
            break

    # ==========================================
    # 🔻 공격적 재탐색 실패 → 가장 좋은 fallback 경로 반환
    # ==========================================
    if best_attempt_poly is None:
        return {
            "status": "error",
            "message": "정밀 경로 탐색 실패 (fallback 후보 없음)",
            "start": start_point_dict,
            "polyline": [start_point_dict],
            "turns": [],
            "summary": {
                "length_m": 0.0,
                "km_requested": km,
                "estimated_time_min": 0.0,
                "turn_count": 0,
            },
            "meta": {},
        }

    # fallback route
    turns, summary = build_turn_by_turn(best_attempt_poly, km_requested=km)

    best_attempt_meta["message"] = (
        f"12회 재탐색에도 정밀 경로를 찾지 못해, 가장 인접한 경로({summary['length_m']}m)를 반환합니다."
    )

    formatted_poly = _format_polyline_for_frontend(best_attempt_poly)

    return {
        "status": "ok",
        "start": start_point_dict,
        "polyline": formatted_poly,
        "turns": turns,
        "summary": summary,
        "meta": best_attempt_meta,
    }
