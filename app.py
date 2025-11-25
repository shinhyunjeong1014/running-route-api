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
    """
    러닝 루프 + 턴바이턴 안내 생성 API.
    - status = "ok"    : 정상 경로
    - status = "error" : 경로 생성 실패 (프론트에서 안내 필요)
    """
    # 1) 러닝 루프 생성 (Valhalla 도보 경로 기반 Area-Loop)
    polyline, meta = generate_area_loop(lat, lng, km)

    # polyline: List[Tuple[lat, lng]] → 프론트에서 쓰기 좋게 dict로 변환
    poly_points = [{"lat": p[0], "lng": p[1]} for p in polyline]

    # 기본 summary/turns 초기화
    turns = []
    summary = {
        "length_m": 0.0,
        "km_requested": km,
        "estimated_time_min": 0.0,
        "turn_count": 0,
    }

    target_m = km * 1000.0
    length_m = float(meta.get("len", 0.0))
    success_flag = bool(meta.get("success", False))

    # -----------------------------
    # 안전성 검증
    # -----------------------------
    # 1) 경로 포인트가 너무 적으면 명백한 실패
    if len(poly_points) < 4:
        return {
            "status": "error",
            "message": "이 위치에서는 러닝 루프를 생성하지 못했습니다. 출발 위치를 조금 이동해서 다시 시도해주세요.",
            "start": {"lat": lat, "lng": lng},
            "polyline": [],
            "turns": [],
            "summary": summary,
            "meta": meta,
        }

    # 2) 거리 비율 검증 (0.6 ~ 1.6배 범위 밖이면 '기괴한 경로/실패'로 간주)
    length_ok = (0.6 * target_m <= length_m <= 1.6 * target_m)

    if not success_flag or not length_ok:
        # polyline은 디버깅/지도 확인용으로 내려주고,
        # 프론트에서는 status="error" 기반으로 사용자 안내.
        return {
            "status": "error",
            "message": "안전한 러닝 루프를 찾지 못했습니다. 출발 위치를 조금 바꾸거나 거리를 조정해보세요.",
            "start": {"lat": lat, "lng": lng},
            "polyline": poly_points,
            "turns": [],
            "summary": summary,
            "meta": meta,
        }

    # -----------------------------
    # 여기까지 통과하면 "정상 루프"로 간주
    # → 턴바이턴 정보 생성
    # -----------------------------
    turns, summary = build_turn_by_turn(poly_points, km_requested=km)

    return {
        "status": "ok",
        "start": {"lat": lat, "lng": lng},
        "polyline": poly_points,
        "turns": turns,
        "summary": summary,
        "meta": meta,
    }
