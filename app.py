from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# 경로 알고리즘 import
from route_algo import generate_running_route
from turn_algo import build_turn_by_turn

app = FastAPI()
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------
# 러닝 루프 + 턴바이턴 API
# ---------------------------------------------------------
@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 위도"),
    lng: float = Query(..., description="시작 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
    quality_first: bool = Query(True, description="품질 우선 모드"),
):
    """
    러닝 루프 생성 + 턴바이턴 결합 API
    """

    try:
        # --------------------------
        # 1) 루프 생성
        # --------------------------
        route_result = generate_running_route(lat, lng, km, quality_first=quality_first)

        if route_result.get("status") != "ok":
            return JSONResponse(content=route_result, status_code=200)

        poly = route_result["polyline"]
        polyline_xy = [(p["lat"], p["lng"]) for p in poly]

        # --------------------------
        # 2) 턴바이턴 생성
        # --------------------------
        turns, summary = build_turn_by_turn(polyline_xy, km)

        # --------------------------
        # 3) 응답 패키징
        # --------------------------
        response = {
            "status": "ok",
            "message": route_result.get("message", "ok"),
            "start": route_result["start"],
            "polyline": poly,
            "distance_km": route_result["distance_km"],
            "turns": turns,
            "summary": summary,
        }

        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        logger.exception(f"[recommend-route] internal error: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "message": "서버 내부 오류가 발생했습니다.",
                "detail": str(e),
            },
            status_code=500,
        )