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

    # -------------------------------
    # 2km 이상 → Valhalla(v1)
    # -------------------------------
    if km >= 2.0:
        result = generate_route_v1(lat, lng, km)

        # v1 polyline = list of dict → convert to tuple list for turn_algo
        polyline_tuples = [(p["lat"], p["lng"]) for p in result["polyline"]]

        # turn-by-turn
        turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)

        result["turns"] = turns
        result["summary"] = summary
        return result

    # -------------------------------
    # 2km 미만 → OSM(v2)
    # -------------------------------
    else:
        polyline, meta = generate_route_v2(lat, lng, km)

        result = {
            "status": "ok",
            "message": meta.get("message", "근거리 경로 생성 완료"),
            "start": {"lat": lat, "lng": lng},
            "polyline": [{"lat": a, "lng": b} for (a, b) in polyline],
            "meta": meta,
        }

        # v2 polyline = tuple list already
        turns, summary = build_turn_by_turn(polyline, km_requested=km)
        result["turns"] = turns
        result["summary"] = summary
        return result