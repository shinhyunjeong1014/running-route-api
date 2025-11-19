from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv
import os, random
import osmnx as ox
from datetime import datetime
import json
from pathlib import Path

from route_algo import (
    build_walk_graph,
    make_loop_route,
    nodes_to_latlngs,
    path_length,
)

load_dotenv()
KAKAO_JS_KEY = os.getenv("KAKAO_JS_KEY", "").strip()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not KAKAO_JS_KEY:
        print("[WARN] KAKAO_JS_KEY 가 .env 에 설정되지 않았습니다.")
    return templates.TemplateResponse("index.html", {"request": request, "kakao_js_key": KAKAO_JS_KEY})


@app.get("/api/recommend-route")
def recommend_route(
    lat: float = Query(..., description="위도"),
    lng: float = Query(..., description="경도"),
    km:  float = Query(..., ge=0.2, le=5.0, description="목표 루프 거리(km)"),
):
    try:
        # 1) 그래프 구성
        G = build_walk_graph(lat, lng, km)
        if len(G) == 0:
            raise RuntimeError("보행 네트워크를 불러오지 못했습니다.")

        # 2) 시작 노드
        try:
            s_node = ox.nearest_nodes(G, lng, lat)
        except Exception:
            # 좌표가 그래프 경계 바깥인 극소수 상황 보완
            s_node = min(G.nodes, key=lambda n: (G.nodes[n]["y"]-lat)**2 + (G.nodes[n]["x"]-lng)**2)

        # 3) 루프 생성 시도
        fallback_used = False
        try:
            nodes, length_m = make_loop_route(G, s_node, km)
        except Exception as e:
            # 4) 실패 시에도 반드시 “닫힌 경로”를 반환 (임시 왕복 루프)
            print(f"[WARN] make_loop_route 실패: {e}")
            fallback_used = True
            nodes = [s_node]
            # 시작점 주변을 몇 번 왕복하여 닫힌 루프 구성(시연용)
            steps = max(6, int(km * 8))  # km 비례로 조금 길게
            cur = s_node
            for _ in range(steps):
                # 이웃 중 하나 선택
                neigh = list(G.neighbors(cur))
                if not neigh:
                    break
                nxt = random.choice(neigh)
                nodes.append(nxt)
                cur = nxt
            # 되돌아오기
            nodes.append(s_node)
            length_m = path_length(G, nodes)

        # 5) 시각 좌표로 변환 (시작/끝 동일 보장)
        coords = nodes_to_latlngs(G, nodes)
        if coords and (coords[0]["lat"] != coords[-1]["lat"] or coords[0]["lng"] != coords[-1]["lng"]):
            coords = coords + [coords[0]]

        payload = {
            "start": {"lat": lat, "lng": lng},
            "km_requested": km,
            "length_m": round(length_m, 1),
            "polyline": coords,
            "meta": {
                "fallback": fallback_used,
                "tolerance_m": 30.0,
                "length_constraint_m": 400.0
            }
        }

        # ▼▼ 자동 저장
        save_dir = Path(__file__).parent / "json"
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"route_{ts}_{km:.1f}km.json"
        with open(save_dir / fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return JSONResponse(payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {e}")
