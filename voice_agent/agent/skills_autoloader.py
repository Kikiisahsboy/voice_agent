# -*- coding: utf-8 -*-
"""Skills 自动加载器 — 通过反射扫描指定模块中的 skill_* 全局变量。"""

import importlib
import inspect
import logging
import os
from typing import Optional

from voice_agent.agent.skill_manager import Skill, SkillManager

logger = logging.getLogger(__name__)


def autoload_skills(
    skill_manager: SkillManager,
    module_paths: list[str],
    extra_dirs: Optional[list[str]] = None,
) -> int:
    """从配置的模块列表中加载所有 Skill 实例。

    扫描规则：模块中所有以 'skill_' 开头的全局变量，如果它是 Skill 实例就注册。
    重复的 skill 不会重复注册。

    Args:
        skill_manager: SkillManager 实例
        module_paths: 模块完整路径列表
        extra_dirs: 额外扫描的目录（深度 1）

    Returns:
        新加载的 skill 数量
    """
    loaded = 0
    seen: set[str] = set()
    registered: set[str] = set(skill_manager.get_skill_names())

    def _scan_module(mod) -> int:
        n = 0
        for name, obj in vars(mod).items():
            if not name.startswith("skill_"):
                continue
            if not isinstance(obj, Skill):
                continue
            if obj.name in registered:
                continue
            skill_manager.register(obj)
            registered.add(obj.name)
            n += 1
        return n

    for mod_path in module_paths:
        if mod_path in seen:
            continue
        seen.add(mod_path)
        try:
            mod = importlib.import_module(mod_path)
            loaded += _scan_module(mod)
        except Exception as e:
            logger.warning("无法导入 skill 模块 '%s': %s", mod_path, e)

    if extra_dirs:
        for d in extra_dirs:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if not fname.endswith("_skill.py"):
                    continue
                mod_name = fname[:-3]
                full_mod = f"{d.replace('/', '.').rstrip('.')}.{mod_name}"
                if full_mod in seen:
                    continue
                seen.add(full_mod)
                try:
                    mod = importlib.import_module(full_mod)
                    loaded += _scan_module(mod)
                except Exception as e:
                    logger.warning("无法导入 skill 模块 '%s': %s", full_mod, e)

    return loaded