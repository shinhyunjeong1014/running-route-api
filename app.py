from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from route_algo import generate_area_loop_a1

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/recommend-route")
def recommend_route(lat: float, lng: float, km: float):
    """러닝 경로 추천 API (도보 전용 + A1 Round 루프)"""
    result = generate_area_loop_a1(lat, lng, km)
    return result
