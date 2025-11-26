from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import logging

# route_algo.py에서 MAX_TOTAL_CALLS를 가져올 필요는 없으나,
# 전체 로직에 영향을 미치지 않도록 주의
from route_algo import generate_area_loop, polyline_length_m
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

    target_m = km * 1000.0
    length_m = meta.get("len", polyline_length_m(polyline))
    
    # 2) 기본적인 생성 실패 또는 너무 짧은 polyline
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

    # 3) 길이/품질 검증 및 Fallback 경로 처리
    
    # 길이 적합성: (0.6 * target_m) ~ (1.6 * target_m) 범위는 유지
    length_ok_for_frontend = 0.6 * target_m <= length_m <= 1.6 * target_m

    # Valhalla 호출에 성공했거나, Fallback 경로가 길이를 만족하는 경우
    is_valid_route = meta.get("success", False) or (meta.get("used_fallback", False) and meta.get("length_ok", False))
    
    # Fallback 경로인 경우, 추가로 턴바이턴 검사 (필수 요구사항)
    if is_valid_route and meta.get("used_fallback"):
        turns, summary = build_turn_by_turn(polyline, km_requested=km)
        
        # 의미 있는 턴 (U턴, 좌/우회전) 최소 1개 검사
        meaningful_turns = [t for t in turns if t["type"] in ("uturn", "left", "right")]
        
        if not meaningful_turns:
            # Fallback 경로가 단순 직선인 경우, 실패 처리
            meta["message"] = "Fallback 경로가 너무 단순하여 안내를 제공할 수 없습니다."
            meta["validation_turns"] = len(meaningful_turns)
            is_valid_route = False

    if not is_valid_route or not length_ok_for_frontend:
        # 실패/비정상 루트 (길이 부적합 포함)
        
        # Fallback 검증 실패 시 message 업데이트
        error_message = meta.get(
            "message",
            "안전한 러닝 루프를 찾지 못했습니다. 출발 위치를 조금 바꾸거나 거리를 조정해 보세요.",
        )
        
        return {
            "status": "error",
            "message": error_message,
            "start": {"lat": lat, "lng": lng},
            # 디버깅/프론트엔드 표시를 위해 polyline 은 전달
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

    # 4) 턴바이턴/요약 생성 (정상 루프 또는 검증된 Fallback 경로)
    # Fallback 경로의 경우 이미 위에서 계산되었을 수 있음.
    if 'summary' not in locals():
        turns, summary = build_turn_by_turn(polyline, km_requested=km)

    return {
        "status": "ok",
        "start": {"lat": lat, "lng": lng},
        "polyline": polyline,
        "turns": turns,
        "summary": summary,
        "meta": meta,
    }
