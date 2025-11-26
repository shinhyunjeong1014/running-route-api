from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# generate_loop_route -> generate_area_loop 로 변경하여 route_algo.py와 통일
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


@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 지점 위도"),
    lng: float = Query(..., description="시작 지점 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
):
    """러닝 루프 추천 API. (응답 구조 유지)"""
    
    start_point = {"lat": lat, "lng": lng}
    
    # 1) 루프 생성 및 메타 정보 획득
    polyline, meta = generate_area_loop(lat, lng, km)

    target_m = km * 1000.0
    length_m = meta.get("len", polyline_length_m(polyline))
    
    # 2) 턴바이턴 정보 생성
    turns, summary = build_turn_by_turn(polyline, km_requested=km)

    
    # 3) 엄격한 경로 검증
    is_successful_route = meta.get("success", False) # 최적 루프 생성 성공

    # Fallback 경로 검증: 
    #   1. route_algo 내에서 길이 검증(± 300m) 통과(meta["length_ok"] == True)
    #   2. turn_algo 검사에서 최소 1개 이상의 의미 있는 턴(uturn, left, right)이 존재
    is_valid_fallback = False
    if meta.get("used_fallback", False) and meta.get("length_ok", False):
        meaningful_turns = [t for t in turns if t["type"] in ("uturn", "left", "right")]
        if meaningful_turns:
            is_valid_fallback = True
        else:
            # Fallback이 너무 단순하거나 직선이어서 의미있는 턴이 없는 경우 실패 처리
            meta["message"] = "Fallback 경로가 너무 단순하여 안내를 제공할 수 없습니다."
            meta["validation_turns"] = len(meaningful_turns)
            
    
    # 4) 최종 상태 결정 및 응답
    
    # 성공 조건: 최적 루프를 찾았거나 (is_successful_route), 유효한 Fallback 경로인 경우
    if is_successful_route or is_valid_fallback:
        # 성공/유효 경로 반환 (FastAPI 응답 형식 유지)
        return {
            "status": "ok",
            "start": start_point,
            "polyline": polyline,
            "turns": turns,
            "summary": summary,
            "meta": meta,
        }
    else:
        # 실패 조건: 최적 루프/유효 Fallback 경로 모두 실패
        # polyline이 아예 없거나, 길이 검증/턴 검증에 실패
        
        # 에러 발생 시, message, polyline, meta 를 항상 포함 (필수 요구사항)
        default_error_message = "경로를 생성하지 못했거나 유효성 검사를 통과하지 못했습니다. 위치/거리를 조정해 보세요."
        
        return {
            "status": "error",
            "message": meta.get("message", default_error_message),
            "start": start_point,
            # 폴리라인이 비어있으면 빈 배열 반환, 아니면 디버깅 위해 현재 폴리라인 반환
            "polyline": polyline if polyline and len(polyline) > 1 else [start_point], 
            "turns": [],
            "summary": {
                "length_m": round(length_m, 1),
                "km_requested": km,
                "estimated_time_min": 0.0,
                "turn_count": 0,
            },
            "meta": meta,
        }
