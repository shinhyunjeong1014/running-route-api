"""
app.py

FastAPI 엔드포인트 정의

GET /api/recommend-route
    - 쿼리 파라미터:
        lat (float) : 시작 위도
        lng (float) : 시작 경도
        km  (float) : 목표 거리 (예: 2.0)

    - 응답 JSON:
        {
          "start": { "lat": ..., "lng": ... },
          "polyline": [ { "lat": ..., "lng": ... }, ... ],
          "turns": [ { ... }, ... ],
          "summary": { ... },
          "meta": "AreaLoop-A1 ..."
        }
"""

from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from route_algo import generate_loop_route
from turn_algo import build_turn_by_turn


app = FastAPI(title="SafeWalk Running Route API")


@app.get("/api/recommend-route")
async def recommend_route(
    lat: float = Query(..., description="시작 위도"),
    lng: float = Query(..., description="시작 경도"),
    km: float = Query(2.0, gt=0.0, description="목표 러닝 거리 (km)"),
):
    """
    러닝용 루프 경로를 생성해 반환한다.
    """
    # 1) 루프 경로 생성 (도보 only, Area-Loop A1)
    polyline, total_length_m, meta = generate_loop_route(lat, lng, km)

    # 2) Turn-by-Turn 이벤트 생성
    turns, summary = build_turn_by_turn(polyline, km_requested=km, total_length_m=total_length_m)

    result = {
        "start": {"lat": lat, "lng": lng},
        "polyline": polyline,
        "turns": turns,
        "summary": summary,
        "meta": meta,
    }
    return JSONResponse(content=result)


@app.get("/")
async def root():
    return {"status": "ok", "message": "SafeWalk Running Route API"}
