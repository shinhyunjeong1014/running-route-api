# app.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import sys
import os

# 현재 경로에 route_algo.py와 turn_algo.py가 있다고 가정하고 임시로 경로 추가
# from route_algo import generate_route
# from turn_algo import build_turn_by_turn

# FastAPI는 로컬에서 실행되므로, 실제 모듈 구조를 가정하고 경로 설정을 시도합니다.
# 이 코드는 실행 환경에 따라 수정이 필요할 수 있습니다.
# Canvas 환경에서는 동일 디렉토리로 간주하고 임포트합니다.

try:
    from route_algo import generate_route
    from turn_algo import build_turn_by_turn
except ImportError:
    # 모듈을 찾지 못할 경우의 대안 (실제 환경에 맞게 수정 필요)
    print("Warning: Failed to import route_algo or turn_algo. Ensure all files are in the same directory.")
    sys.exit(1)


app = FastAPI(
    title="FastLoopRoute API v4",
    description="러닝 앱을 위한 안정적인 루프 경로 추천 서비스"
)

@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 위도 (예: 37.5665)"),
    lng: float = Query(..., description="시작 경도 (예: 126.9780)"),
    km: float = Query(2.0, ge=1.0, le=10.0, description="목표 거리 (km, 1.0~10.0)")
):
    """
    시작 좌표와 목표 거리를 기반으로 루프 형태의 경로를 추천합니다.
    """
    try:
        # 1. 경로 생성 (route_algo.py)
        polyline, length_m, algorithm_used = generate_route(lat, lng, km)
        
        # 2. Turn-by-turn 및 요약 생성 (turn_algo.py)
        turns, summary = build_turn_by_turn(polyline, km_requested=km, total_length_m=length_m)

        return JSONResponse({
            "start": {"lat": lat, "lng": lng},
            "polyline": polyline,
            "turns": turns,
            "summary": summary,
            "meta": {
                "generation": f"FastLoopRoute v4 ({algorithm_used})",
                "tolerance_m": 250, # 2km 기준 1.8km~2.3km
                "algorithm_used": algorithm_used
            }
        }, status_code=200)

    except RuntimeError as e:
        # 경로 생성 알고리즘에서 의도적으로 발생시킨 에러
        return JSONResponse({"error": str(e), "message": "경로 생성에 실패했습니다. 주변 네트워크 밀도가 너무 낮거나 요청 범위가 비현실적일 수 있습니다."}, status_code=404)
    except Exception as e:
        # 기타 서버 에러
        return JSONResponse({"error": str(e), "message": "서버 내부 오류가 발생했습니다."}, status_code=500)
