from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import List, Dict, Tuple, Any

# route_algo와 turn_algo가 같은 디렉토리에 있다고 가정
# 주의: generate_area_loop가 generate_running_route로 변경되었으며,
# polyline_length_m은 이제 generate_running_route 내부에서 처리됩니다.
from route_algo import generate_running_route
from turn_algo import build_turn_by_turn

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

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


def _format_polyline_for_frontend(
    polyline_dicts: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    """
    route_algo에서 반환된 [{"lat": lat, "lng": lng}, ...] 형태를 그대로 반환합니다.
    (route_algo가 이미 포맷팅하여 반환하도록 변경됨)
    """
    return polyline_dicts


@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="시작 지점 위도"),
    lng: float = Query(..., description="시작 지점 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
):
    """러닝 루프 추천 API. (유효 경로가 있을 때만 status: ok 반환)"""
    
    start_point_dict = {"lat": lat, "lng": lng}
    
    # 1) 루프 생성 및 결과 획득 (통합된 generate_running_route 사용)
    # generate_running_route는 모든 결과(상태, 폴리라인, 메타)를 단일 딕셔너리로 반환합니다.
    route_result: Dict[str, Any] = generate_running_route(lat, lng, km)
    
    status = route_result.get("status", "error")
    polyline_dicts = route_result.get("polyline", [])
    distance_km = route_result.get("distance_km", 0.0)
    
    # Valhalla 기반 로직은 모든 메타 정보를 root에 포함하고,
    # OSMnx 기반 로직은 'meta' 키 안에 추가 정보를 포함합니다.
    # 여기서는 모든 메타 정보를 통일하여 'meta' 키로 정리합니다.
    meta = {
        "message": route_result.get("message", "경로 생성 시도 완료"),
        "distance_km": distance_km,
    }
    # OSMnx 로직에서 추가로 제공되는 'meta' 정보를 병합
    if 'meta' in route_result:
        meta.update(route_result['meta'])
    
    
    # 2) 유효성 확인: 상태가 'ok'이고 폴리라인이 있을 때만 유효 경로로 간주
    is_valid_route = status == "ok" and len(polyline_dicts) > 1 and distance_km > 0.001

    # 3) 경로가 유효할 때만 턴바이턴 정보 생성 및 응답
    if is_valid_route:
        
        # 턴바이턴을 위해 List[Dict[str, float]] 형태를 List[Tuple[float, float]]로 역변환
        polyline_tuples: List[Tuple[float, float]] = [
            (p['lat'], p['lng']) for p in polyline_dicts
        ]
        
        # 턴바이턴 정보 생성
        # build_turn_by_turn 함수가 km_requested 대신 distance_km을 사용해야 할 수 있습니다.
        # 여기서는 기존과 같이 km_requested를 넘깁니다.
        turns, summary = build_turn_by_turn(polyline_tuples, km_requested=km)
        
        # message 재설정 (Valhalla 로직의 메시지를 기본으로 사용)
        final_message = route_result.get("message", "경로 생성 완료")
        
        # summary 업데이트 (summary는 turn_algo에서 계산된 최종값 사용)
        summary['length_m'] = round(distance_km * 1000.0, 3)
        summary['km_requested'] = km
        
        meta["message"] = final_message
        
        # 3.1. 유효 경로가 있을 경우: status: ok 반환
        return {
            "status": "ok",
            "start": start_point_dict,
            "polyline": polyline_dicts, # 이미 포맷된 리스트 사용
            "turns": turns,
            "summary": summary,
            "meta": meta,
        }
    else:
        # 3.2. 경로 생성에 실패했을 경우: status: error 반환
        
        length_m = round(distance_km * 1000.0, 3)
        
        return {
            "status": "error",
            "message": route_result.get("message", "탐색 결과, 유효한 경로 후보를 찾을 수 없습니다."),
            "start": start_point_dict,
            "polyline": [start_point_dict], # 0m 경로 (시작점 하나)
            "turns": [],
            "summary": {"length_m": length_m, "km_requested": km, "estimated_time_min": 0.0, "turn_count": 0},
            "meta": meta,
        }
