# -*- coding: utf-8 -*-
"""Skill 注册中心 — 管理所有可被 LLM 调用的技能。

支持依赖注入：handler 可接收 `_context: dict` 参数，由 orchestrator 注入。
"""

import inspect
import json
import logging
from dataclasses import dataclass
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
        if skill.name in self._skills:
            logger.warning("Skill '%s' 已存在，将被覆盖", skill.name)
        self._skills[skill.name] = skill
        logger.info("注册 Skill: %s", skill.name)

    def unregister(self, name: str):
        """移除一个 skill。"""
        self._skills.pop(name, None)

    def get_all_tools(self) -> list[dict]:
        """将所有已注册 skill 转为 Ollama tool calling 格式。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            }
            for s in self._skills.values()
        ]

    def execute(self, name: str, args: dict, context: dict | None = None) -> str:
        """根据名称和参数执行 skill，返回结果字符串。

        Args:
            name: skill 名称
            args: 参数（dict 或 JSON 字符串）
            context: 注入到 handler 的依赖（如 {"memory": mm}）
        """
        skill = self._skills.get(name)
        if not skill:
            return f"错误：未找到技能 '{name}'"

        try:
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            elif args is None:
                args = {}

            sig = inspect.signature(skill.handler)
            params = sig.parameters

            # 如果 handler 接受 **_kwargs，把所有 args 当 kwargs 传
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            # 找出声明的具名参数（不含 _context）
            declared = [
                p for p in params.values()
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                and p.name not in ("_context", "self")
            ]

            call_kwargs = {}

            if has_var_kw or len(declared) > 0:
                # 只把 handler 实际声明的参数填进去
                for p in declared:
                    if p.name in args:
                        call_kwargs[p.name] = args[p.name]
                    elif p.default is inspect.Parameter.empty:
                        # 必填参数缺失
                        return f"错误：技能 '{name}' 缺少必填参数 '{p.name}'"
                if has_var_kw:
                    # 也允许额外参数
                    for k, v in args.items():
                        if k not in call_kwargs:
                            call_kwargs[k] = v

                if "_context" in params:
                    call_kwargs["_context"] = context or {}

                result = skill.handler(**call_kwargs) if call_kwargs else skill.handler()
            else:
                # handler 不接受任何业务参数（只接受 _context）
                if "_context" in params:
                    result = skill.handler(_context=context or {})
                else:
                    result = skill.handler()

            return str(result)
        except Exception as e:
            logger.error("Skill '%s' 执行失败: %s", name, e, exc_info=True)
            return f"执行技能 '{name}' 时出错: {e}"

    def get_skill_names(self) -> list[str]:
        return list(self._skills.keys())

    def get_skill_descriptions(self) -> str:
        """获取所有 skill 的描述文本，用于 system prompt。"""
        return "\n".join(f"- {s.name}: {s.description}" for s in self._skills.values())