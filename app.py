from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import List, Dict, Tuple

# 새 route_algo.py (Part1~4) 기준
from route_algo import generate_running_route

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
# 헬스체크: 서버 살아있는지 확인
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
    (스파이크 제거 + 품질 기반 루프 생성)
    """

    # route_algo.py의 generate_running_route()는
    # 이미 status / polyline / distance_km / message를 포함한 dict를 반환함
    result = generate_running_route(lat, lng, km, quality_first=quality_first)

    # 형식 그대로 클라이언트에게 전달
    return JSONResponse(content=result)