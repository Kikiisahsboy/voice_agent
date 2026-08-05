# -*- coding: utf-8 -*-
"""时间查询 Skill。"""

from datetime import datetime

from voice_agent.agent.skill_manager import Skill


def _get_time(format: str = "datetime") -> str:
    now = datetime.now()
    if format == "time":
        return now.strftime("%H:%M:%S")
    elif format == "date":
        return now.strftime("%Y年%m月%d日")
    else:
        return now.strftime("%Y年%m月%d日 %H:%M:%S")


skill_time = Skill(
    name="get_current_time",
    description="获取当前的系统日期和时间。当用户询问'现在几点'、'今天几号'、'当前时间'时使用此技能。",
    parameters={
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["time", "date", "datetime"],
                "description": "返回格式：time=仅时间, date=仅日期, datetime=日期+时间",
            }
        },
        "required": [],
    },
    handler=_get_time,
    category="utility",
)
