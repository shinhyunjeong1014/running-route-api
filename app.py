from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from typing import List, Dict, Tuple

# route_algo와 turn_algo가 같은 디렉토리에 있다고 가정
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
    """ [[lat, lng], ...] 형태를 [{"lat":lat, "lng":lng}, ...] 형태로 변환 """
    return [{"lat": lat, "lng": lng} for lat, lng in polyline]


@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 지점 위도"),
    lng: float = Query(..., description="시작 지점 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
):
    """
    러닝 루프 추천 API
    - 요청거리 이상 ~ 요청거리 +99m 이내를 success(True)로 판단
    - 실패 시 최대 5회까지 재탐색 반복
    """
    start_point_dict = {"lat": lat, "lng": lng}

    MAX_RETRY = 5
    RETRY_DELAY = 0.1  # 0.1초 대기

    best_attempt_poly = None
    best_attempt_meta = None

    # ==========================================
    # 🔄 1) 재탐색 루프
    # ==========================================
    for attempt in range(1, MAX_RETRY + 2):  # 첫 시도 + 5회 재시도
        polyline_tuples, meta = generate_area_loop(lat, lng, km)

        is_valid_route = (
            polyline_tuples
            and len(polyline_tuples) >= 2
            and polyline_length_m(polyline_tuples) > 0
        )

        # 기록(가장 인접한 것을 fallback으로 남기기)
        if is_valid_route:
            # 첫 valid route는 fallback 후보로 저장
            best_attempt_poly = polyline_tuples
            best_attempt_meta = meta

            # success=True 면 즉시 return
            if meta.get("success", False):
                turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)
                final_message = (
                    "요청 거리보다 0~99m 이내로 긴 정밀 경로가 도출되었습니다."
                )
                meta["message"] = final_message

                formatted_poly = _format_polyline_for_frontend(polyline_tuples)

                return {
                    "status": "ok",
                    "start": start_point_dict,
                    "polyline": formatted_poly,
                    "turns": turns,
                    "summary": summary,
                    "meta": meta,
                }

        # success=False → 재탐색
        if attempt <= MAX_RETRY:
            time.sleep(RETRY_DELAY)
            continue
        else:
            break

    # ==========================================
    # 🔻 여기 도달한 경우 = MAX_RETRY까지 success 경로 못 찾음
    # ==========================================

    # fallback: 가장 인접한 경로도 못 찾은 극단적 경우
    if best_attempt_poly is None:
        length_m = best_attempt_meta.get("len", 0.0) if best_attempt_meta else 0.0
        return {
            "status": "error",
            "message": "정밀 경로 탐색 실패 (fallback 루트 없음)",
            "start": start_point_dict,
            "polyline": [start_point_dict],
            "turns": [],
            "summary": {
                "length_m": length_m,
                "km_requested": km,
                "estimated_time_min": 0.0,
                "turn_count": 0,
            },
            "meta": best_attempt_meta,
        }

    # fallback route 반환 (success=False지만 인접 경로 존재)
    turns, summary = build_turn_by_turn(best_attempt_poly, km_requested=km)

    best_attempt_meta["message"] = (
        best_attempt_meta.get(
            "message",
            f"요청 거리 이상 0~99m 이내의 정밀 경로를 찾지 못해, 가장 인접한 경로({summary['length_m']}m)를 반환합니다."
        )
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
