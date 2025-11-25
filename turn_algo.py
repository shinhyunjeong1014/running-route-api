# turn_algo.py
import math
import logging
from route_algo import haversine_m, _cumulative_distances 
# route_algo.py의 haversine_m, _cumulative_distances 함수를 사용합니다.

# 회전 감지 임계값
ANGLE_UTURN = 150   # U턴 임계각 (150도 이상)
ANGLE_TURN = 25     # 좌/우회전 임계각 (25도 이상)
MIN_DIST_TURN = 35  # 최소 회전 감지 거리 (m)
MIN_DIST_SIMPLIFY = 15.0 # Polyline 간소화 최소 거리 (m)
STRAIGHT_INTERVAL = 300 # 직진 안내 추가 간격 (m)

###############################################
# Polyline Simplification (경로 간소화)
###############################################
def simplify_polyline(polyline):
    """
    Douglas-Peucker 알고리즘은 사용하지 않고, 
    최소 거리 임계값(MIN_DIST_SIMPLIFY)을 이용해 불필요한 노드를 제거합니다.
    """
    if not polyline or len(polyline) < 3:
        return polyline
    
    simp = [polyline[0]]
    last = polyline[0]
    
    for p in polyline[1:]:
        d = haversine_m(last["lat"], last["lng"], p["lat"], p["lng"])
        if d >= MIN_DIST_SIMPLIFY:
            simp.append(p)
            last = p
            
    # 마지막 지점이 포함되지 않았다면 추가
    if simp[-1] != polyline[-1]:
        simp.append(polyline[-1])
        
    logging.debug(f"경로 간소화: 원본 {len(polyline)}개 -> 간소화 {len(simp)}개")
    return simp


###############################################
# Angles / Bearings (방위각 및 회전각 계산)
###############################################
def bearing(a, b):
    """두 지점 사이의 방위각(Bearing)을 0~360도로 계산합니다."""
    lat1, lon1 = math.radians(a["lat"]), math.radians(a["lng"])
    lat2, lon2 = math.radians(b["lat"]), math.radians(b["lng"])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1)*math.cos(lat2) - math.sin(lat1)*math.sin(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def signed_angle(a, b, c):
    """b 지점에서의 회전각을 -180도(우회전) ~ +180도(좌회전)로 계산합니다."""
    th1 = bearing(a, b) # A->B 방향
    th2 = bearing(b, c) # B->C 방향
    # 두 방위각의 차이를 -180 ~ +180 범위로 변환
    return ((th2 - th1 + 540) % 360) - 180


###############################################
# Text formatting for instruction (안내 문구 형식)
###############################################
def format_instruction(distance_m, turn_type):
    """거리와 회전 유형에 따라 안내 문구를 생성합니다."""
    # 거리를 10m 단위로 반올림
    d = int(round(distance_m / 10.0) * 10)
    
    if turn_type == "left":
        return f"{d}m 앞에서 좌회전하세요"
    if turn_type == "right":
        return f"{d}m 앞에서 우회전하세요"
    if turn_type == "straight":
        return f"{d}m 직진하세요"
    if turn_type == "uturn":
        return f"{d}m 앞에서 U턴하세요"
    if turn_type == "arrive":
        return "목적지에 도착했습니다. 러닝을 완료했습니다."
    if turn_type == "start":
        return "러닝을 시작합니다. 직진하세요."
    return ""


###############################################
# Main turn detection (회전 감지 로직)
###############################################
def polyline_turns(polyline):
    """간소화된 경로를 기반으로 Turn-by-turn 안내를 생성합니다."""
    if not polyline or len(polyline) < 3:
        return []
    
    simp = simplify_polyline(polyline)
    cum = _cumulative_distances(simp) # 간소화된 경로의 누적 거리

    turns = []
    last_turn_idx = 0
    
    # 0. 시작 안내
    turns.append({
        "lat": simp[0]["lat"],
        "lng": simp[0]["lng"],
        "type": "start",
        "distance": 0.0,
        "instruction": format_instruction(0, "start"),
    })
    
    last_turn_dist = 0.0
    
    # 1. 회전 감지 및 안내 생성
    for i in range(1, len(simp) - 1):
        a, b, c = simp[i-1], simp[i], simp[i+1]
        ang = signed_angle(a, b, c)
        ang_abs = abs(ang)

        dist_to_turn = cum[i] - last_turn_dist
        t_type = None

        # A. 회전 감지
        if ang_abs >= ANGLE_UTURN:
            t_type = "uturn"
        elif ang_abs >= ANGLE_TURN:
            t_type = "left" if ang > 0 else "right"
        
        # B. 직진 안내 강제 삽입 (단조로운 경로 보정)
        # 회전이 감지되지 않았고, 마지막 안내로부터 일정 거리 이상 지났을 때 직진 안내 추가
        if not t_type and dist_to_turn >= STRAIGHT_INTERVAL:
             # 임시로 직진 안내를 삽입할 지점의 직전 노드를 사용
             # 실제 turn은 다음 노드(i)에서 발생하므로 dist_to_turn은 cum[i]를 사용
             t_type = "straight"
        
        
        # C. 유효한 안내가 생성된 경우
        if t_type and dist_to_turn >= MIN_DIST_TURN:
            turns.append({
                "lat": b["lat"],
                "lng": b["lng"],
                "type": t_type,
                "angle": round(ang, 1),
                "distance": round(dist_to_turn, 1),
                "instruction": format_instruction(dist_to_turn, t_type),
            })
            last_turn_dist = cum[i] # 마지막 안내가 발생한 누적 거리 업데이트
        
        # 직진 안내가 강제 삽입되었다면, 다음 회전을 위해 t_type을 리셋하지 않고 다음 반복으로 이동
        if t_type == "straight":
             # 직진 안내가 삽입된 경우, 다음 감지는 직진 안내 지점(b)에서 다시 시작
             last_turn_dist = cum[i] 


    # 2. 마지막 도착 안내
    final_dist = cum[-1] - last_turn_dist
    turns.append({
        "lat": simp[-1]["lat"],
        "lng": simp[-1]["lng"],
        "type": "arrive",
        "distance": round(final_dist, 1),
        "instruction": format_instruction(final_dist, "arrive"),
    })

    return turns


def build_turn_by_turn(polyline, km_requested: float, total_length_m: float):
    """Turn-by-turn 안내와 요약 정보를 생성합니다."""

    # 1. Turn-by-turn 안내 생성
    turns = polyline_turns(polyline)
    
    # 2. Summary 생성
    # 걷는 속도: 4km/h = 66.67 m/min
    # 러닝 속도: 8km/h = 133.33 m/min (8.0 min/km)
    RUNNING_SPEED_KMH = 8.0
    
    estimated_time_min = (total_length_m / 1000) / (RUNNING_SPEED_KMH / 60) # 총 거리 / (km/min)
    
    summary = {
        "length_m": round(total_length_m, 1),
        "km_requested": km_requested,
        "estimated_time_min": round(estimated_time_min, 1),
        "turn_count": len([t for t in turns if t["type"] in ["left", "right", "uturn", "straight"]])
    }

    return turns, summary
