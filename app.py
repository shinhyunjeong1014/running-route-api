from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from route_algo import generate_area_loop, polyline_length_m
from turn_algo import build_turn_by_turn


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


@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 지점 위도"),
    lng: float = Query(..., description="시작 지점 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
):
    """러닝 루프 추천 API.

    1. Valhalla 기반 루프 polyline 생성
    2. 거리/품질 검증
    3. 턴바이턴 정보 및 요약 반환
    """
    # 1) 루프 생성
    polyline, meta = generate_area_loop(lat, lng, km)

    if not polyline or len(polyline) < 2:
        return {
            "status": "error",
            "message": meta.get(
                "message",
                "경로를 생성하지 못했습니다. 출발 위치를 조금만 옮겨 다시 시도해 주세요.",
            ),
            "start": {"lat": lat, "lng": lng},
            "polyline": [],
            "turns": [],
            "summary": {
                "length_m": 0.0,
                "km_requested": km,
                "estimated_time_min": 0.0,
                "turn_count": 0,
            },
            "meta": meta,
        }

    # 2) 거리/품질 검증
    target_m = km * 1000.0
    length_m = meta.get("len")
    if length_m is None:
        length_m = polyline_length_m(polyline)
        meta["len"] = length_m

    # 너무 짧거나 긴 루프는 프론트에서 '이상 경로'로 처리할 수 있도록 status=error 로 보냄
    length_ok = 0.6 * target_m <= length_m <= 1.6 * target_m

    if not meta.get("success", False) or not length_ok:
        # 실패/비정상 루트지만, 디버깅을 위해 polyline 은 같이 전달
        return {
            "status": "error",
            "message": meta.get(
                "message",
                "안전한 러닝 루프를 찾지 못했습니다. 출발 위치를 조금 바꾸거나 거리를 조정해 보세요.",
            ),
            "start": {"lat": lat, "lng": lng},
            "polyline": polyline,
            "turns": [],
            "summary": {
                "length_m": round(length_m, 1),
                "km_requested": km,
                "estimated_time_min": 0.0,
                "turn_count": 0,
            },
            "meta": meta,
        }

    # 3) 턴바이턴/요약 생성
    turns, summary = build_turn_by_turn(polyline, km_requested=km)

    return {
        "status": "ok",
        "start": {"lat": lat, "lng": lng},
        "polyline": polyline,
        "turns": turns,
        "summary": summary,
        "meta": meta,
    }
