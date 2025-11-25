# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from route_algo import generate_area_loop
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

    polyline, meta = generate_area_loop(lat, lng, km)

    turns, summary = build_turn_by_turn(
        [{"lat": p[0], "lng": p[1]} for p in polyline],
        km_requested=km,
    )

    return {
        "start": {"lat": lat, "lng": lng},
        "polyline": [{"lat": p[0], "lng": p[1]} for p in polyline],
        "turn_by_turn": turns,
        "summary": summary,
        "meta": meta,
    }
