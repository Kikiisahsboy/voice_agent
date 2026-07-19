# -*- coding: utf-8 -*-
"""Skill 注册中心 — 管理所有可被 LLM 调用的技能。"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any]
    category: str = "general"


class SkillManager:
    """Skill 注册和路由。"""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        """注册一个 skill。"""
        self._skills[skill.name] = skill
        logger.info("注册 Skill: %s", skill.name)

    def unregister(self, name: str):
        """移除一个 skill。"""
        self._skills.pop(name, None)

    def get_all_tools(self) -> list[dict]:
        """将所有已注册 skill 转为 Ollama tool calling 格式。"""
        tools = []
        for skill in self._skills.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": skill.name,
                    "description": skill.description,
                    "parameters": skill.parameters,
                },
            })
        return tools

    def execute(self, name: str, args: dict) -> str:
        """根据名称和参数执行 skill，返回结果字符串。"""
        skill = self._skills.get(name)
        if not skill:
            return f"错误：未找到技能 '{name}'"

        try:
            if isinstance(args, str):
                import json
                args = json.loads(args)
            result = skill.handler(**args) if args else skill.handler()
            return str(result)
        except Exception as e:
            logger.error("Skill '%s' 执行失败: %s", name, e)
            return f"执行技能 '{name}' 时出错: {e}"

    def get_skill_names(self) -> list[str]:
        return list(self._skills.keys())

    def get_skill_descriptions(self) -> str:
        """获取所有 skill 的描述文本，用于 system prompt。"""
        lines = []
        for skill in self._skills.values():
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)
