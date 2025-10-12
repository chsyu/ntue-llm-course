"""
簡化的天氣工具
"""
import requests
import json
from langchain_core.tools import tool

def _get_location(city: str):
    """獲取城市座標"""
    url = "https://geopy-server.vercel.app/city_to_lat_lng_timezone/"
    params = {"city": city}  # 交給 requests 自動編碼
    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    if not data:
        raise ValueError(f"找不到城市: {city}")

    return data["latitude"], data["longitude"], data["city"]

@tool
def get_current_weather(city: str) -> str:
    """獲取指定城市的當前天氣狀況，包括溫度、濕度、風速等即時資訊"""
    try:
        lat, lon, name = _get_location(city)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m"
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        current = data["current"]
        
        result = {
            "城市": name,
            "溫度": f"{current['temperature_2m']}°C",
            "天氣": _weather_code(current['weather_code']),
            "風速": f"{current['wind_speed_10m']}km/h",
            "濕度": f"{current['relative_humidity_2m']}%"
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        return f"獲取天氣失敗: {e}"


@tool  
def get_weather_forecast(city: str, days: int = 7) -> str:
    """獲取指定城市的天氣預報，支援1-16天的未來天氣趨勢查詢
    
    Args:
        city: 城市名稱
        days: 預報天數，預設7天，最多支援16天
    """
    try:
        # 限制在 1-16 天之間，預設為 7 天
        days = max(1, min(days, 16))
        lat, lon, name = _get_location(city)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "forecast_days": days
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        daily = data["daily"]
        
        forecast = []
        for i in range(len(daily["time"])):
            forecast.append({
                "日期": daily["time"][i],
                "最高溫": f"{daily['temperature_2m_max'][i]}°C",
                "最低溫": f"{daily['temperature_2m_min'][i]}°C",
                "天氣": _weather_code(daily['weather_code'][i])
            })
        
        result = {"城市": name, "預報": forecast}
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        return f"獲取預報失敗: {e}"

def _weather_code(code: int) -> str:
    """簡化的天氣代碼"""
    codes = {
        0: "晴朗", 1: "晴朗", 2: "多雲", 3: "陰天",
        45: "霧", 48: "霧", 51: "小雨", 53: "中雨", 55: "大雨",
        61: "雨", 63: "雨", 65: "大雨", 71: "雪", 73: "雪", 75: "大雪",
        80: "陣雨", 81: "陣雨", 82: "大陣雨", 95: "雷暴"
    }
    return codes.get(code, "未知")