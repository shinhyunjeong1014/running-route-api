from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import List, Dict, Tuple

# 반드시 generate_running_route 사용해야 함
from route_algo import generate_running_route, polyline_length_m
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
    return [{"lat": lat, "lng": lng} for lat, lng in polyline]


@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 지점 위도"),
    lng: float = Query(..., description="시작 지점 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
):
    start_point_dict = {"lat": lat, "lng": lng}

    # ⭐ 중요: 이제 generate_running_route 호출해야 함 (2km 분기 포함)
    result = generate_running_route(lat, lng, km)

    polyline_tuples = [(p["lat"], p["lng"]) for p in result["polyline"]]
    dist_m = polyline_length_m(polyline_tuples)

    is_valid_route = polyline_tuples and dist_m > 0

    if is_valid_route:
        turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)

        result["turns"] = turns
        result["summary"] = summary

        return result

    else:
        return {
            "status": "error",
            "message": result.get("message", "유효한 경로를 찾을 수 없습니다."),
            "start": start_point_dict,
            "polyline": [start_point_dict],
            "turns": [],
            "summary": {"length_m": 0, "km_requested": km, "estimated_time_min": 0.0},
            "meta": result.get("meta", {}),
        }