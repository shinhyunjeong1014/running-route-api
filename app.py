from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import List, Dict, Tuple

# 모듈 import
from route_algo import generate_running_route
from turn_algo import build_turn_by_turn

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

app = FastAPI()

# ---------------------------------------------------------
# CORS 설정
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)


# ---------------------------------------------------------
# 헬스체크
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------
# recommend-route API
# ---------------------------------------------------------
@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 지점 위도"),
    lng: float = Query(..., description="시작 지점 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
    quality_first: bool = Query(True, description="품질 우선 모드"),
):
    """
    러닝 루프 추천 API
    1. route_algo: 루프 생성 및 스파이크 제거
    2. turn_algo: 턴바이턴 안내 및 시간 예측 생성
    """
    
    # 1. 경로 생성 (route_algo)
    # 반환 형식: {'status':..., 'polyline': [{'lat':.., 'lng':..}], ...}
    result = generate_running_route(lat, lng, km, quality_first=quality_first)

    # 경로 생성 실패 시 바로 반환
    if result.get("status") != "ok":
        return JSONResponse(content=result)

    try:
        # 2. 데이터 변환 (List[Dict] -> List[Tuple])
        # turn_algo는 [(lat, lng), ...] 튜플 리스트를 입력으로 받음
        raw_poly = result.get("polyline", [])
        poly_tuples = [(p["lat"], p["lng"]) for p in raw_poly]

        # 3. 턴바이턴 안내 생성 (turn_algo)
        turns, summary = build_turn_by_turn(poly_tuples, km_requested=km)

        # 4. 결과 병합
        # 기존 result에 안내 정보(turns)와 요약 정보(summary) 추가
        result["turns"] = turns
        result["summary"] = summary
        
        # (선택 사항) 편의를 위해 summary의 예상 시간을 상위 레벨에도 노출 가능
        result["estimated_time_min"] = summary.get("estimated_time_min", 0)

    except Exception as e:
        logger.error(f"Turn generation failed: {e}")
        # 턴 생성에 실패하더라도 경로는 반환 (turns는 빈 리스트)
        result["turns"] = []
        result["summary"] = {}

    return JSONResponse(content=result)
