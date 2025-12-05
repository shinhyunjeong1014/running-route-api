from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List, Dict, Tuple
import os
import pickle
import time

# 수정된 모듈 import (route_algo, turn_algo는 이전 버전 유지)
from route_algo import generate_area_loop, polyline_length_m
from turn_algo import build_turn_by_turn_async

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# ============================
# Global Graph Storage
# ============================
global_graph = None
MAP_FILE = "my_area.pickle"  # Pickle 파일 사용

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----------------------------------------------------
    # 서버 시작 시: Pickle 데이터 로드 (고속)
    # ----------------------------------------------------
    global global_graph
    
    print("\n" + "="*50)
    print("🚀 서버 시작 프로세스 가동")
    print("="*50)

    try:
        if os.path.exists(MAP_FILE):
            print(f"📂 맵 파일({MAP_FILE}) 발견! 메모리로 로드합니다...")
            start_time = time.time()
            
            # [핵심] Pickle 로드: 파싱 과정 없이 메모리에 바로 적재됨 (매우 빠름)
            with open(MAP_FILE, "rb") as f:
                global_graph = pickle.load(f)
                
            elapsed = time.time() - start_time
            print(f"✅ 맵 로드 완료! (소요시간: {elapsed:.2f}초)")
            print(f"📍 로드된 노드 개수: {len(global_graph.nodes)}개")
            print("✨ 서버 준비 완료! 요청을 받을 수 있습니다.\n")
            
        else:
            print(f"❌ 오류: '{MAP_FILE}' 파일이 없습니다!")
            print("👉 먼저 'python init_map.py'를 실행해서 맵 파일을 만들어주세요.")
            global_graph = None
            
    except Exception as e:
        print(f"❌ 맵 로드 중 치명적 오류 발생: {e}")
        global_graph = None

    yield
    
    # 서버 종료 시 정리
    print("🛑 서버 종료: 메모리를 정리합니다.")
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
    """서버 상태 및 맵 로드 여부 확인"""
    return {
        "status": "ok", 
        "map_ready": global_graph is not None,
        "map_nodes": len(global_graph.nodes) if global_graph else 0
    }


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
    1. Pre-loaded Graph (Memory) 사용 -> I/O 대기 없음
    2. Async POI 검색 -> Network 대기 최소화
    """
    start_point_dict = {"lat": lat, "lng": lng}
    
    # 1) 맵 데이터 준비 확인
    if global_graph is None:
        return {
            "status": "error",
            "message": "서버에 맵 데이터가 로드되지 않았습니다. 관리자에게 문의하세요.",
            "start": start_point_dict,
            "polyline": [start_point_dict],
            "turns": [],
            "summary": {"length_m": 0, "km_requested": km, "estimated_time_min": 0, "event_count": 0},
            "meta": {"success": False}
        }

    # 2) 루프 생성 (CPU 연산)
    # route_algo.py는 이미 Graph 객체를 받도록 수정되었음
    polyline_tuples, meta = generate_area_loop(global_graph, lat, lng, km)
    
    is_valid_route = polyline_tuples and polyline_length_m(polyline_tuples) > 0

    if is_valid_route:
        # 3) [Await] 비동기 턴바이턴 생성 (I/O 병렬 처리)
        turns, summary = await build_turn_by_turn_async(polyline_tuples, km_requested=km)
        
        # 메시지 처리
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

