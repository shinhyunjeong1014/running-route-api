from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from route_algo import (
    get_graph_cached,
    generate_loop_route,
    nodes_to_latlngs,
    build_turn_by_turn,
    path_length
)

app = FastAPI()


@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 위도"),
    lng: float = Query(..., description="시작 경도"),
    km: float = Query(..., ge=0.3, le=5.0, description="목표 루프 거리 (km)")
):
    try:
        # 1) 캐싱된 그래프 로딩 (빠름)
        G, s_node = get_graph_cached(lat, lng)

        # 2) 삼각 루프 기반 경로 생성 (FastLoopRoute v2)
        nodes = generate_loop_route(G, s_node, km)

        # 3) 폴리라인 좌표로 변환
        coords = nodes_to_latlngs(G, nodes)

        # 4) 경로 길이 산출
        length_m = path_length(G, nodes)

        # 5) turn-by-turn 안내 생성
        turns, summary = build_turn_by_turn(coords, km_requested=km, total_length_m=length_m)

        return JSONResponse({
            "start": {"lat": lat, "lng": lng},
            "polyline": coords,
            "turns": turns,
            "summary": summary,
            "meta": {
                "generation": "FastLoopRoute v2",
                "tolerance_m": 150.0
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
