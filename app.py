from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
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


def _format_polyline_for_frontend(polyline: List[Tuple[float, float]]) -> List[Dict[str, float]]:
    """
    (lat, lng) 튜플 리스트를 프론트(Vue/Flutter/html)에서 사용하기 쉬운
    {lat: ..., lng: ...} 딕셔너리 리스트로 변환.
    """
    return [{"lat": float(lat), "lng": float(lng)} for lat, lng in polyline]


@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 지점 위도"),
    lng: float = Query(..., description="시작 지점 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
):
    """
    러닝 루프 추천 API.

    1) route_algo.generate_area_loop()를 호출해 polyline + meta 생성
    2) turn_algo.build_turn_by_turn()을 호출해 턴바이턴 안내 정보 생성
    3) runner_map.html 등에서 바로 사용할 수 있는 JSON 형태로 반환
    """
    logger.info(f"[recommend-route] lat={lat}, lng={lng}, km={km}")

    # 1. 러닝 루프 생성
    polyline, meta = generate_area_loop(lat, lng, km)

    # polyline 길이(m) 재계산 (meta["len"]이 있더라도 안전하게 보정)
    length_m = polyline_length_m(polyline)
    meta["len"] = float(length_m)

    # polyline이 비었는지 여부
    has_route = bool(polyline) and len(polyline) >= 2

    # 시작점 정보
    start_point_dict = {"lat": float(lat), "lng": float(lng)}

    # 프론트에서 사용하는 polyline 포맷으로 변환
    polyline_tuples: List[Tuple[float, float]] = [(float(p[0]), float(p[1])) for p in polyline]
    polyline_for_front = _format_polyline_for_frontend(polyline_tuples)

    # ==========================
    # 2. 경로가 존재하는 경우
    # ==========================
    if has_route:
        # 2.1. 턴바이턴 안내 생성
        # - build_turn_by_turn(polyline, km_requested=km) 사용
        try:
            turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)
        except Exception as e:
            logger.exception(f"build_turn_by_turn 실패: {e}")
            # 턴 정보를 만들지 못해도, 일단 polyline은 반환
            turns = []
            summary = {
                "length_m": length_m,
                "km_requested": km,
                "estimated_time_min": 0.0,
                "turn_count": 0,
            }

        # summary에 실제 길이 보정
        summary["length_m"] = float(length_m)

        # 메타 정보 보강
        meta["turn_count"] = summary.get("turn_count", len(turns))
        meta["summary_length_m"] = summary["length_m"]
        meta["km_requested"] = km

        # success 여부에 따라 message 조정
        is_valid_route = meta.get("success", False)

        if is_valid_route:
            # 턴바이턴 정보 생성 (len > 0 보장)
            turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)

            # message 재설정
            if meta.get("success", False):
                # ±99m 완벽 충족
                final_message = "최적의 정밀 경로가 도출되었습니다."
            else:
                # ±99m 초과, 하지만 가장 인접한 경로를 반환함
                final_message = meta.get(
                    "message",
                    f"요청 오차(±99m)를 초과하지만, 가장 인접한 경로({summary['length_m']}m)를 반환합니다.",
                )

            meta["message"] = final_message

        # 최종 응답
        response = {
            "status": "ok",
            "message": meta.get("message", "루프를 생성했습니다."),
            "start": start_point_dict,
            "polyline": polyline_for_front,
            "turns": turns,
            "summary": summary,
            "meta": meta,
        }
        return JSONResponse(content=response)

    # ==========================
    # 3. 경로가 없는 경우 (fallback 포함)
    # ==========================
    else:
        # 3.1. fallback 사각형 루프를 사용했는데도 polyline이 1개 이하인 경우
        # 또는 meta["success"]가 False 인데, 사실상 사용할 수 없는 경우
        # => status: error 처리
        if not polyline_for_front or len(polyline_for_front) < 2:
            length_m = meta.get("len", 0.0)
            return {
                "status": "error",
                "message": meta.get("message", "탐색 결과, 유효한 경로 후보를 찾을 수 없습니다."),
                "start": start_point_dict,
                "polyline": [start_point_dict],  # 0m 경로 (시작점 하나)
                "turns": [],
                "summary": {
                    "length_m": length_m,
                    "km_requested": km,
                    "estimated_time_min": 0.0,
                    "turn_count": 0,
                },
                "meta": meta,
            }

        # 3.2. 경로 후보를 0개 찾았을 경우 (len=0): status: error 반환
        length_m = meta.get("len", 0.0)

        return {
            "status": "error",  # [핵심] status: error 반환
            "message": meta.get("message", "탐색 결과, 유효한 경로 후보를 찾을 수 없습니다."),
            "start": start_point_dict,
            "polyline": [start_point_dict],  # 0m 경로 (시작점 하나)
            "turns": [],
            "summary": {
                "length_m": length_m,
                "km_requested": km,
                "estimated_time_min": 0.0,
                "turn_count": 0,
            },
            "meta": meta,
        }
