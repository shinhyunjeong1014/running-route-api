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


def _format_polyline_for_frontend(
    polyline: List[Tuple[float, float]],
) -> List[Dict[str, float]]:
    """
    [[lat, lng], ...] 형태를 [{"lat": lat, "lng": lng}, ...] 형태로 변환합니다.
    """
    return [{"lat": lat, "lng": lng} for lat, lng in polyline]


@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 지점 위도"),
    lng: float = Query(..., description="시작 지점 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
):
    """러닝 루프 추천 API. (유효 경로가 있을 때만 status: ok 반환)"""
    
    start_point_dict = {"lat": lat, "lng": lng}
    
    # 1) 루프 생성 및 메타 정보 획득 (route_algo에서 모든 최적화 수행)
    polyline_tuples, meta = generate_area_loop(lat, lng, km)
    
    # 2) 유효성 확인: polyline_tuples의 길이가 2개 이상이고, 실제 길이가 0m를 초과할 때만 유효 경로로 간주
    is_valid_route = polyline_tuples and polyline_length_m(polyline_tuples) > 0

    # 3) 경로가 유효할 때만 턴바이턴 정보 생성 및 응답
    if is_valid_route:
        
        # 턴바이턴 정보 생성 (len > 0 보장)
        turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)
        
        # message 재설정
        if meta.get("success", False):
            # ±99m 완벽 충족
            final_message = "최적의 정밀 경로가 도출되었습니다."
        else:
            # ±99m 초과, 하지만 가장 인접한 경로를 반환함
            final_message = meta.get("message", f"요청 오차(±99m)를 초과하지만, 가장 인접한 경로({summary['length_m']}m)를 반환합니다.")

        meta["message"] = final_message
        
        formatted_polyline = _format_polyline_for_frontend(polyline_tuples)

        # 3.1. 유효 경로가 있을 경우: status: ok 반환
        return {
            "status": "ok",
            "start": start_point_dict,
            "polyline": formatted_polyline,
            "turns": turns,
            "summary": summary,
            "meta": meta,
        }
    else:
        # 3.2. 경로 후보를 0개 찾았을 경우 (len=0): status: error 반환
        
        length_m = meta.get("len", 0.0)
        
        return {
            "status": "error", # [핵심] status: error 반환
            "message": meta.get("message", "탐색 결과, 유효한 경로 후보를 찾을 수 없습니다."),
            "start": start_point_dict,
            "polyline": [start_point_dict], # 0m 경로 (시작점 하나)
            "turns": [],
            "summary": {"length_m": length_m, "km_requested": km, "estimated_time_min": 0.0, "turn_count": 0},
            "meta": meta,
        }
