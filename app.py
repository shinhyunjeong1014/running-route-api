from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Tuple

# 2km 이상 전용
from route_algo_v1 import generate_running_route as generate_route_v1

# 2km 미만 전용
from route_algo_v2 import generate_area_loop as generate_route_v2
from route_algo_v2 import polyline_length_m

from turn_algo import build_turn_by_turn


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)


@app.get("/api/recommend-route")
def recommend_route(
    lat: float,
    lng: float,
    km: float,
):
    # --------------------------------------
    # 엄격한 분기
    # --------------------------------------
    if km >= 2.0:
        # 첫 번째 코드 (Valhalla)
        result = generate_route_v1(lat, lng, km)
        polyline = result["polyline"]  # [{lat, lng}, ...]

    else:
        # 두 번째 코드 (OSMNX)
        polyline, meta = generate_route_v2(lat, lng, km)
        result = {
            "status": "ok",
            "message": meta.get("message", "근거리 경로 생성 완료"),
            "start": {"lat": lat, "lng": lng},
            "polyline": [{"lat": a, "lng": b} for (a, b) in polyline],
            "meta": meta,
        }

    # turn-by-turn 생성
    poly_as_tuple = [(p["lat"], p["lng"]) for p in result["polyline"]]
    turns, summary = build_turn_by_turn(poly_as_tuple, km_requested=km)

    result["turns"] = turns
    result["summary"] = summary

    return result