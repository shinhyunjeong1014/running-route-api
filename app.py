from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
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
    """러닝 루프 추천 API. (최우선 경로 추천을 위한 검증 완화)"""
    
    start_point_dict = {"lat": lat, "lng": lng}
    
    # 1) 루프 생성
    polyline_tuples, meta = generate_area_loop(lat, lng, km)

    # 2) 턴바이턴 정보 생성
    turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)

    # 3) 엄격한 경로 검증
    is_successful_route = meta.get("success", False)

    is_valid_fallback = False
    if meta.get("used_fallback", False) and meta.get("length_ok", False):
        # [수정] Fallback 경로에 대한 '최소 턴 개수' 검증 로직 제거 -> 길이 OK면 무조건 유효
        is_valid_fallback = True
            
    
    # 4) 최종 상태 결정 및 응답
    
    formatted_polyline = _format_polyline_for_frontend(polyline_tuples)
    
    if is_successful_route or is_valid_fallback:
        # Fallback도 무조건 OK 처리 (최우선 경로 추천)
        return {
            "status": "ok",
            "start": start_point_dict,
            "polyline": formatted_polyline, 
            "turns": turns,
            "summary": summary,
            "meta": meta,
        }
    else:
        # 여전히 경로를 찾지 못한 경우 (Valhalla 완전 실패)
        default_error_message = "경로를 생성하지 못했거나 유효성 검사를 통과하지 못했습니다. 위치/거리를 조정해 보세요."
        
        summary = {
            "length_m": 0.0,
            "km_requested": km,
            "estimated_time_min": 0.0,
            "turn_count": 0,
        }
        
        return {
            "status": "error",
            "message": meta.get("message", default_error_message),
            "start": start_point_dict,
            "polyline": formatted_polyline if polyline_tuples and len(polyline_tuples) > 1 else [start_point_dict], 
            "turns": [],
            "summary": summary,
            "meta": meta,
        }
