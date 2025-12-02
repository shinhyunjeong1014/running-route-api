import requests
import json

# 인천 연수·남동·미추홀 구역 bbox 통합
bbox = "37.40,126.60,37.50,126.78"

query = f"""
[out:json][timeout:50];
(
  way["building"="apartments"]({bbox});
  relation["building"="apartments"]({bbox});

  way["building"="residential"]({bbox});
  relation["building"="residential"]({bbox});

  way["landuse"="residential"]({bbox});
  relation["landuse"="residential"]({bbox});
);
(._;>;);
out geom;
"""

url = "https://overpass-api.de/api/interpreter"
resp = requests.post(url, data={"data": query})
data = resp.json()

with open("redzones.geojson", "w") as f:
    json.dump(data, f)

print("Saved redzones.geojson")
