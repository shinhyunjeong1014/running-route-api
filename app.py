from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
# ▼ [변경 1] 기본 JSONResponse 대신 속도가 20~50배 빠른 ORJSONResponse 사용
from fastapi.responses import ORJSONResponse
# ▼ [변경 2] 데이터 압축 전송을 위한 GZipMiddleware 사용
from fastapi.middleware.gzip import GZipMiddleware

import logging
from typing import List, Dict, Tuple
import uvicorn

# route_algo와 turn_algo는 기존 파일 그대로 사용
from route_algo import generate_area_loop, polyline_length_m
from turn_algo import build_turn_by_turn

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# ▼ [변경 3] default_response_class를 ORJSONResponse로 설정하여 모든 응답 속도 향상
app = FastAPI(default_response_class=ORJSONResponse)

# ▼ [변경 4] 1KB 이상 데이터는 자동으로 압축하여 전송 (네트워크 대기 시간 단축)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

def _format_polyline_for_frontend(
    polyline: List[Tuple[float, float]],
) -> List[Dict[str, float]]:
    return [{"lat": lat, "lng": lng} for lat, lng in polyline]

# ▼ [중요 체크] async def가 아닌 'def'를 유지해야 합니다.
# async def로 바꾸면 연산하는 1분 동안 서버 전체가 멈춥니다. 
# def로 두면 FastAPI가 별도 스레드에서 실행하여 서버 멈춤을 방지합니다.
@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 지점 위도"),
    lng: float = Query(..., description="시작 지점 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
):
    """러닝 루프 추천 API."""
    
    start_point_dict = {"lat": lat, "lng": lng}
    
    # 1) 루프 생성 (시간이 오래 걸리는 작업)
    polyline_tuples, meta = generate_area_loop(lat, lng, km)
    
    # 2) 유효성 확인
    is_valid_route = polyline_tuples and polyline_length_m(polyline_tuples) > 0

    if is_valid_route:
        # 턴바이턴 정보 생성
        turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)
        
        if meta.get("success", False):
            final_message = "최적의 정밀 경로가 도출되었습니다."
        else:
            final_message = meta.get("message", f"요청 오차(±99m)를 초과하지만, 가장 인접한 경로({summary['length_m']}m)를 반환합니다.")

        meta["message"] = final_message
        formatted_polyline = _format_polyline_for_frontend(polyline_tuples)

        return {
            "status": "ok",
            "start": start_point_dict,
            "polyline": formatted_polyline,
            "turns": turns,
            "summary": summary,
            "meta": meta,
        }
    else:
        length_m = meta.get("len", 0.0)
        return {
            "status": "error",
            "message": meta.get("message", "탐색 결과, 유효한 경로 후보를 찾을 수 없습니다."),
            "start": start_point_dict,
            "polyline": [start_point_dict],
            "turns": [],
            "summary": {"length_m": length_m, "km_requested": km, "estimated_time_min": 0.0, "turn_count": 0},
            "meta": meta,
        }

if __name__ == "__main__":
    # ▼ [변경 5] 타임아웃 설정을 300초(5분)로 넉넉하게 설정
    # 이렇게 해야 1분 연산 중에도 연결이 끊기지 않습니다.
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        timeout_keep_alive=300
    )


