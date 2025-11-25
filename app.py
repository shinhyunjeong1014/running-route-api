# app.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import sys

try:
    from route_algo import generate_route
    from turn_algo import build_turn_by_turn
except ImportError:
    print("Error: Failed to import route_algo or turn_algo. Ensure all files are in the same directory.")
    sys.exit(1)

app = FastAPI(
    title="FastLoopRoute API v5",
    description="러닝 앱을 위한 안정적인 루프 경로 추천 서비스 - Fallback 강화 버전"
)

@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 위도 (예: 37.5665)"),
    lng: float = Query(..., description="시작 경도 (예: 126.9780)"),
    km: float = Query(2.0, ge=1.0, le=10.0, description="목표 거리 (km, 1.0~10.0)")
):
    """
    시작 좌표와 목표 거리를 기반으로 루프 형태의 경로를 추천합니다.
    OSM 네트워크 부족 시에도 자동으로 Fallback 경로를 생성합니다.
    """
    try:
        # 1. 경로 생성 (route_algo.py) - 절대 실패하지 않음
        polyline, length_m, algorithm_used = generate_route(lat, lng, km)
        
        # 2. Turn-by-turn 및 요약 생성 (turn_algo.py)
        turns, summary = build_turn_by_turn(polyline, km_requested=km, total_length_m=length_m)

        return JSONResponse({
            "start": {"lat": lat, "lng": lng},
            "polyline": polyline,
            "turns": turns,
            "summary": summary,
            "meta": {
                "generation": f"FastLoopRoute v5 ({algorithm_used})",
                "tolerance_m": 250,
                "algorithm_used": algorithm_used
            }
        }, status_code=200)

    except Exception as e:
        # 모든 예외를 캐치하여 서버 오류로 반환 (경로 생성은 항상 성공해야 함)
        return JSONResponse({
            "error": str(e), 
            "message": "예상치 못한 오류가 발생했습니다. 다시 시도해주세요."
        }, status_code=500)
