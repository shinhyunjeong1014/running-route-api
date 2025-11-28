# route_algo.py — B안: Start → viaA → viaB → Start 구조로 안정적 루프 생성
# ============================================================

import math
import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
import requests

logger = logging.getLogger("route_algo")
logger.setLevel(logging.INFO)

VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002/route")
VALHALLA_TIMEOUT = float(os.environ.get("VALHALLA_TIMEOUT", "2.5"))
VALHALLA_MAX_RETRY = 2
GLOBAL_TIMEOUT = 10.0
MAX_TOTAL_CALLS = 30
MAX_LENGTH_ERROR = 99.0

KAKAO_API_KEY = "dc3686309f8af498d7c62bed0321ee64"
KAKAO_ROUTE_URL = "https://apis-navi.kakaomobility.com/v1/directions"


# ------------------------------------------------------
# 기본 함수
# ------------------------------------------------------

def haversine_m(a_lat, a_lng, b_lat, b_lng):
    R = 6371000.0
    p1 = math.radians(a_lat)
    p2 = math.radians(b_lat)
    dphi = p2 - p1
    dl = math.radians(b_lng - a_lng)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*(math.sin(dl/2)**2)
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))


def polyline_length(points):
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (la1, lo1), (la2, lo2) in zip(points, points[1:]):
        total += haversine_m(la1, lo1, la2, lo2)
    return total


def project_point(lat, lng, dist_m, bearing_deg):
    R = 6371000.0
    br = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lam1 = math.radians(lng)
    phi2 = math.asin(math.sin(phi1)*math.cos(dist_m/R) +
                     math.cos(phi1)*math.sin(dist_m/R)*math.cos(br))
    lam2 = lam1 + math.atan2(math.sin(br)*math.sin(dist_m/R)*math.cos(phi1),
                             math.cos(dist_m/R) - math.sin(phi1)*math.sin(phi2))
    return math.degrees(phi2), (math.degrees(lam2) + 540) % 360 - 180


# ------------------------------------------------------
# Valhalla / Kakao API
# ------------------------------------------------------

def valhalla_route(p1: Tuple[float, float], p2: Tuple[float, float]) -> List[Tuple[float, float]]:
    lat1, lon1 = p1
    lat2, lon2 = p2

    costing = {
        "pedestrian": {
            "sidewalk_preference": 1.0,
            "private_road_penalty": 5000,
            "avoid_steps": 1.0
        }
    }

    last_err = None
    for _ in range(VALHALLA_MAX_RETRY):
        try:
            payload = {
                "locations": [
                    {"lat": lat1, "lon": lon1, "type": "break"},
                    {"lat": lat2, "lon": lon2, "type": "break"}
                ],
                "costing": "pedestrian",
                "costing_options": costing
            }
            r = requests.post(VALHALLA_URL, json=payload, timeout=VALHALLA_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            shape = data["trip"]["legs"][0]["shape"]
            return decode_polyline(shape)
        except Exception as e:
            last_err = e

    return []


def kakao_route(p1: Tuple[float, float], p2: Tuple[float, float]):
    if not KAKAO_API_KEY:
        return None

    h = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    lon1, lat1 = p1[1], p1[0]
    lon2, lat2 = p2[1], p2[0]
    params = {
        "origin": f"{lon1},{lat1}",
        "destination": f"{lon2},{lat2}",
        "priority": "RECOMMEND",
        "car_model": "walk"
    }

    try:
        r = requests.get(KAKAO_ROUTE_URL, params=params, headers=h, timeout=2.5)
        r.raise_for_status()
        j = r.json()
        if j.get("routes") and j["routes"][0]["result_code"] == 0:
            pts = []
            for sec in j["routes"][0]["sections"]:
                for rd in sec.get("roads", []):
                    vtx = rd.get("vertexes", [])
                    for i in range(0, len(vtx), 2):
                        pts.append((vtx[i+1], vtx[i]))
            if pts:
                if pts[0] != p1:
                    pts.insert(0, p1)
                if pts[-1] != p2:
                    pts.append(p2)
                return pts
    except:
        pass

    return None


def decode_polyline(enc):
    pts = []
    lat = 0
    lng = 0
    idx = 0

    while idx < len(enc):
        result = 1
        shift = 0
        b = 0
        while True:
            b = ord(enc[idx]) - 63
            idx += 1
            result += (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        result = 1
        shift = 0
        while True:
            b = ord(enc[idx]) - 63
            idx += 1
            result += (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        pts.append((lat / 1e6, lng / 1e6))
    return pts


# ------------------------------------------------------
# B안: viaA → viaB 두 개의 Out segment 사용
# ------------------------------------------------------

def generate_two_out_loop(start: Tuple[float, float], target_m: float, start_time: float):
    """
    B안의 핵심:
      Start → viaA → viaB → Start
    Out1 + Out2 + Back >= target_m 를 자연스럽게 보장.
    """

    lat, lng = start

    # Out 전체 길이를 target의 75~85%로 확보
    OUT_RATIO = 0.8
    OUT_TOTAL = target_m * OUT_RATIO  # 약 2400m (3km 기준)

    # Out1, Out2 비율 나누기
    OUT1 = OUT_TOTAL * 0.5     # 약 1200m
    OUT2 = OUT_TOTAL * 0.5     # 약 1200m

    bearings = [0, 45, 90, 135, 180, 225, 270, 315]
    best = None
    best_err = 1e12

    for b1 in bearings:
        viaA = project_point(lat, lng, OUT1, b1)
        out1 = valhalla_route(start, viaA)
        if len(out1) < 2:
            continue

        for b2 in bearings:
            viaB = project_point(viaA[0], viaA[1], OUT2, b2)
            out2 = valhalla_route(viaA, viaB)
            if len(out2) < 2:
                continue

            back = kakao_route(viaB, start) or valhalla_route(viaB, start)
            if len(back) < 2:
                continue

            route = out1 + out2[1:] + back[1:]
            L = polyline_length(route)
            err = abs(L - target_m)

            if err < best_err:
                best_err = err
                best = route

            if time.time() - start_time > GLOBAL_TIMEOUT:
                break

    return best


# ------------------------------------------------------
# Spur extend (길이 부족한 경우 확장)
# ------------------------------------------------------

def extend_with_spur(route, target_m, start_time):
    """
    루프 길이가 target보다 부족한 경우,
    중간 인근에 작은 왕복 스퍼를 붙여 길이를 증가.
    """
    if not route or len(route) < 4:
        return route

    for _ in range(3):
        L = polyline_length(route)
        if abs(L - target_m) <= MAX_LENGTH_ERROR:
            return route

        if L > target_m:
            return route

        # 필요한 추가 거리
        needed = target_m - L
        spur_len = min(needed, 1200)

        idx = len(route) // 2
        p_mid = route[idx]

        # 좌/우 90도 방향으로 spur 후보 만들어봄
        br = 90
        spur_target = project_point(p_mid[0], p_mid[1], spur_len/2, br)
        spur1 = kakao_route(p_mid, spur_target)
        spur2 = kakao_route(spur_target, p_mid)

        if spur1 and spur2:
            spur_path = spur1 + spur2[1:]
            route = route[:idx+1] + spur_path[1:] + route[idx+1:]

        if time.time() - start_time > GLOBAL_TIMEOUT:
            break

    return route


# ------------------------------------------------------
# 최종 메인 함수
# ------------------------------------------------------

def generate_area_loop(lat, lng, km):
    start_time = time.time()
    start = (lat, lng)
    target_m = km * 1000.0

    # 1) B안: 두 개 Out segment로 루프 생성
    route = generate_two_out_loop(start, target_m, start_time)

    if not route:
        return [start], {"error": "루프 생성 실패"}

    # 2) 길이가 부족하면 스퍼로 연장
    route = extend_with_spur(route, target_m, start_time)

    L = polyline_length(route)
    ok = abs(L - target_m) <= MAX_LENGTH_ERROR

    return route, {
        "len": L,
        "err": abs(L - target_m),
        "success": ok,
        "message": "완료" if ok else "목표 거리에 근접한 루프 생성",
    }
