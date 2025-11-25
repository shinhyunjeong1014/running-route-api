# app.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from route_algo import generate_route
from turn_algo import build_turn_by_turn

app = FastAPI()

@app.get("/api/recommend-route")
def recommend_route(lat: float, lng: float, km: float):
    try:
        polyline, length_m = generate_route(lat, lng, km)
        turns, summary = build_turn_by_turn(polyline, km_requested=km, total_length_m=length_m)

        return JSONResponse({
            "start": {"lat": lat, "lng": lng},
            "polyline": polyline,
            "turns": turns,
            "summary": summary,
            "meta": {
                "generation": "FastLoopRoute v2",
                "tolerance_m": 150
            }
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
