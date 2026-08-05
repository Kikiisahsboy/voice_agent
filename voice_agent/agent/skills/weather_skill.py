# -*- coding: utf-8 -*-
"""天气查询 Skill。

默认 mock 实现（返回友好提示）。如需真实数据，
接入 OpenWeatherMap / wttr.in / 心知天气 等，替换 _fetch_weather 即可。
"""

from voice_agent.agent.skill_manager import Skill


def _fetch_weather(city: str) -> str:
    """真实接入示例：使用 wttr.in（无需 API Key）。"""
    try:
        import requests
        resp = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1", "lang": "zh"},
            timeout=8,
        )
        if resp.status_code != 200:
            return f"暂时无法获取 {city} 的天气。"

        data = resp.json()
        current = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]

        city_name = city
        try:
            area_name = area.get("areaName", [{}])[0].get("value", "")
            if area_name:
                city_name = area_name
        except Exception:
            pass

        desc = current.get("lang_zh", [{}])[0].get("value", "") \
            or current.get("weatherDesc", [{}])[0].get("value", "")
        temp = current.get("temp_C", "?")
        feels = current.get("FeelsLikeC", "?")
        humidity = current.get("humidity", "?")

        return (
            f"{city_name}当前天气：{desc}，气温{temp}°C，"
            f"体感{feels}°C，湿度{humidity}%。"
        )
    except Exception as e:
        return f"天气查询失败: {e}"


def _get_weather(city: str) -> str:
    if not city or not city.strip():
        return "请告诉我城市名。"
    return _fetch_weather(city.strip())


skill_weather = Skill(
    name="get_weather",
    description="查询指定城市的当前天气。当用户询问'天气'、'某地下雨吗'、'气温'时使用。",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名，如'北京'、'上海'、'New York'",
            }
        },
        "required": ["city"],
    },
    handler=_get_weather,
    category="utility",
)