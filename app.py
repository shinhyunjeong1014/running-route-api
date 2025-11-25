from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import os
from route_algo import (
    get_graph_cached,
    simple_loop_route,
    nodes_to_latlngs,
    build_turn_by_turn,
    path_length
)

app = FastAPI()

@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(...),
    lng: float = Query(...),
    km: float = Query(..., ge=0.5, le=5.0)
):
    try:
        # 1) 캐싱된 그래프 가져오기 (1초 → 0.05초)
        G, nearest = get_graph_cached(lat, lng)

        # 2) 빠른 루프 생성 (0.1~0.3초)
        nodes = simple_loop_route(G, nearest, km)

        # 3) 좌표 변환
        coords = nodes_to_latlngs(G, nodes)

        # 4) 턴 안내 생성
        length_m = path_length(G, nodes)
        turns, summary = build_turn_by_turn(coords, km_requested=km, total_length_m=length_m)

        return JSONResponse({
            "start": {"lat": lat, "lng": lng},
            "polyline": coords,
            "turns": turns,
            "summary": summary,
            "meta": {
                "fallback": False,
                "generation": "FastLoopRoute v1"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
