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
    # 1) 러닝 루프 생성 (Valhalla 도보 경로 기반 Area-Loop)
    polyline, meta = generate_area_loop(lat, lng, km)

    # polyline: List[Tuple[lat, lng]] → 프론트에서 쓰기 좋게 dict로 변환
    poly_points = [{"lat": p[0], "lng": p[1]} for p in polyline]

    # 2) 턴바이턴 음성 안내용 정보 생성
    turns, summary = build_turn_by_turn(poly_points, km_requested=km)

    return {
        "start": {"lat": lat, "lng": lng},
        "polyline": poly_points,
        "turns": turns,      # 이전 JSON 구조 유지
        "summary": summary,
        "meta": meta,
    }
