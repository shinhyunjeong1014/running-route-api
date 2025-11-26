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
    """러닝 루프 추천 API. (응답 Polyline 구조 및 출발지=도착지 유지)"""
    
    start_point_dict = {"lat": lat, "lng": lng}
    
    # 1) 루프 생성 및 메타 정보 획득 (route_algo에서 이미 루프를 닫았음)
    polyline_tuples, meta = generate_area_loop(lat, lng, km)

    # 2) 턴바이턴 정보 생성 (폐쇄된 루프 기준으로 계산)
    turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)

    # 3) 엄격한 경로 검증
    is_successful_route = meta.get("success", False)

    # Fallback 경로 검증: 길이 OK + 의미 있는 턴 1개 이상
    is_valid_fallback = False
    if meta.get("used_fallback", False) and meta.get("length_ok", False):
        meaningful_turns = [t for t in turns if t["type"] in ("uturn", "left", "right")]
        if meaningful_turns:
            is_valid_fallback = True
        else:
            meta["message"] = "Fallback 경로가 너무 단순하여 안내를 제공할 수 없습니다."
            meta["validation_turns"] = len(meaningful_turns)
            
    
    # 4) 최종 상태 결정 및 응답
    
    # 최종 폴리라인 형식 변환 (프론트엔드 호환)
    formatted_polyline = _format_polyline_for_frontend(polyline_tuples)
    
    if is_successful_route or is_valid_fallback:
        # 성공/유효 경로 반환
        return {
            "status": "ok",
            "start": start_point_dict,
            "polyline": formatted_polyline, 
            "turns": turns,
            "summary": summary,
            "meta": meta,
        }
    else:
        # 실패 조건: 최적 루프/유효 Fallback 경로 모두 실패
        default_error_message = "경로를 생성하지 못했거나 유효성 검사를 통과하지 못했습니다. 위치/거리를 조정해 보세요."
        
        # summary와 length_m을 실패 기준으로 재설정
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
            # 실패 시에도 기존 JSON 형식에 맞춰 변환된 폴리라인 반환
            "polyline": formatted_polyline if polyline_tuples and len(polyline_tuples) > 1 else [start_point_dict], 
            "turns": [],
            "summary": summary,
            "meta": meta,
        }
