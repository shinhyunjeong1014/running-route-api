from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List, Dict, Tuple
import os

try:
    import osmnx as ox
except Exception:
    ox = None

# 수정된 모듈 import
from route_algo import generate_area_loop, polyline_length_m
from turn_algo import build_turn_by_turn_async

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# ============================
# Global Graph Storage
# ============================
global_graph = None

MAP_FILE = "my_area.graphml"
TARGET_AREAS = [
    "Michuhol-gu, Incheon, South Korea",
    "Yeonsu-gu, Incheon, South Korea",
    "Namdong-gu, Incheon, South Korea"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----------------------------------------------------
    # 서버 시작 시: 맵 데이터 로드
    # ----------------------------------------------------
    global global_graph
    logger.info("Initializing Map Data...")
    
    if ox is None:
        logger.error("osmnx is not installed. Map loading skipped.")
        yield
        return

    try:
        # 1. 파일이 있으면 로드
        if os.path.exists(MAP_FILE):
            logger.info(f"Loading from {MAP_FILE}...")
            G = ox.load_graphml(MAP_FILE)
        else:
            # 2. 파일이 없으면 다운로드 (Backup)
            logger.warning(f"Map file '{MAP_FILE}' not found. Downloading areas: {TARGET_AREAS}")
            G = ox.graph_from_place(TARGET_AREAS, network_type="walk", simplify=True)
            # 다음번 실행을 위해 저장
            ox.save_graphml(G, MAP_FILE)
            logger.info(f"Map saved to {MAP_FILE}")
            
        # 3. Undirected 변환 & 메모리 업로드
        global_graph = ox.utils_graph.get_undirected(G)
        logger.info(f"Global Graph loaded! Nodes: {len(global_graph.nodes)}")
        
    except Exception as e:
        logger.error(f"Failed to load map data: {e}")
        global_graph = None

    yield
    # 서버 종료 시 정리
    global_graph = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "map_ready": global_graph is not None}


def _format_polyline_for_frontend(polyline: List[Tuple[float, float]]) -> List[Dict[str, float]]:
    return [{"lat": lat, "lng": lng} for lat, lng in polyline]


@app.get("/api/recommend-route")
async def recommend_route(
    lat: float = Query(..., description="시작 지점 위도"),
    lng: float = Query(..., description="시작 지점 경도"),
    km: float = Query(..., gt=0.1, lt=50.0, description="목표 거리(km)"),
):
    """
    [Async] 러닝 루프 추천 API
    - Pre-loaded Graph 사용으로 즉시 응답
    - Async POI 검색으로 대기시간 최소화
    """
    start_point_dict = {"lat": lat, "lng": lng}
    
    # 1) 메모리 로드된 그래프 전달
    #    (None일 경우 내부에서 Fallback 사각형 루프 작동)
    polyline_tuples, meta = generate_area_loop(global_graph, lat, lng, km)
    
    is_valid_route = polyline_tuples and polyline_length_m(polyline_tuples) > 0

    if is_valid_route:
        # 2) [Await] 비동기 턴바이턴 생성
        turns, summary = await build_turn_by_turn_async(polyline_tuples, km_requested=km)
        
        final_message = meta.get("message", "")
        if meta.get("success", False):
            final_message = "최적의 정밀 경로가 도출되었습니다."
        elif not final_message:
            final_message = f"요청 오차 범위를 초과하지만, 가장 인접한 경로({summary['length_m']}m)를 반환합니다."
        
        meta["message"] = final_message
        
        return {
            "status": "ok",
            "start": start_point_dict,
            "polyline": _format_polyline_for_frontend(polyline_tuples),
            "turns": turns,
            "summary": summary,
            "meta": meta,
        }
    else:
        # 유효 경로 없음 (Error)
        return {
            "status": "error",
            "message": meta.get("message", "탐색 결과, 유효한 경로 후보를 찾을 수 없습니다."),
            "start": start_point_dict,
            "polyline": [start_point_dict],
            "turns": [],
            "summary": {"length_m": meta.get("len", 0.0), "km_requested": km, "estimated_time_min": 0.0, "event_count": 0},
            "meta": meta,
        }
