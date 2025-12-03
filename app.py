from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from route_algo_v1 import generate_running_route as generate_route_v1
from route_algo_v2 import generate_area_loop as generate_route_v2
from turn_algo import build_turn_by_turn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)


@app.get("/api/recommend-route")
def recommend_route(lat: float, lng: float, km: float):

    # ---------------------------------------------------------
    # 2km 이상 → Valhalla 기반 v1
    # ---------------------------------------------------------
    if km >= 2.0:
        result = generate_route_v1(lat, lng, km)

        # polyline이 비었으면 바로 error 리턴
        if not result["polyline"]:
            return result  # v1이 이미 error 구조 반환함

        # dict list → tuple list 변환
        polyline_tuples = [(p["lat"], p["lng"]) for p in result["polyline"]]

        # turn-by-turn
        turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)
        result["turns"] = turns
        result["summary"] = summary
        return result

    # ---------------------------------------------------------
    # 2km 미만 → OSM 기반 v2
    # ---------------------------------------------------------
    else:
        polyline, meta = generate_route_v2(lat, lng, km)

        result = {
            "status": "ok",
            "message": meta.get("message", "근거리 경로 생성 완료"),
            "start": {"lat": lat, "lng": lng},
            "polyline": [{"lat": a, "lng": b} for (a, b) in polyline],
            "meta": meta,
        }

        turns, summary = build_turn_by_turn(polyline, km_requested=km)
        result["turns"] = turns
        result["summary"] = summary
        return result