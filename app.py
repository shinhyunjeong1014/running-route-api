from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse

from route_algo import generate_loop_route
from turn_algo import build_turn_by_turn

app = FastAPI(title="Running Route API with Valhalla")


@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 위도"),
    lng: float = Query(..., description="시작 경도"),
    km: float = Query(..., gt=0.3, le=10.0, description="목표 루프 거리(km)"),
):
    """
    Valhalla 기반 러닝 루프 + turn-by-turn 안내 API
    """
    try:
        polyline, length_m, meta = generate_loop_route(lat, lng, km)

        if not polyline or length_m <= 0:
            raise RuntimeError("경로 생성 실패")

        turns, summary = build_turn_by_turn(
            polyline,
            km_requested=km,
            total_length_m=length_m,
        )

        payload = {
            "start": {"lat": lat, "lng": lng},
            "polyline": polyline,          # [ {lat, lng}, ... ]
            "turns": turns,                # turn-by-turn 안내 리스트
            "summary": summary,            # 총 거리, 예상 시간, turn 개수
            "meta": meta,                  # 엔진/후보정보/디버그
        }
        return JSONResponse(payload)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"경로 생성 중 오류가 발생했습니다: {e}",
        )
