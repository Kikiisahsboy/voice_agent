"""Skills autoloader 测试。"""

import os
import sys

from voice_agent.agent.skill_manager import SkillManager
from voice_agent.agent.skills_autoloader import autoload_skills


def test_autoload_known_modules():
    sm = SkillManager()
    n = autoload_skills(sm, [
        "voice_agent.agent.skills.time_skill",
        "voice_agent.agent.skills.calculator_skill",
        "voice_agent.agent.skills.memory_skill",
    ])
    assert n >= 3
    names = set(sm.get_skill_names())
    assert "get_current_time" in names
    assert "calculate" in names
    assert "search_memory" in names


def test_autoload_missing_module_logs_warning():
    sm = SkillManager()
    n = autoload_skills(sm, ["nonexistent.module"])
    assert n == 0


def test_autoload_idempotent():
    sm = SkillManager()
    n1 = autoload_skills(sm, ["voice_agent.agent.skills.time_skill"])
    n2 = autoload_skills(sm, ["voice_agent.agent.skills.time_skill"])
    assert n1 == 1
    assert n2 == 0  # 已注册，不重复


def test_autoload_picks_only_skill_prefixed():
    """扫描模块时只加载 skill_* 命名的对象。"""
    sm = SkillManager()
    # calculator_skill.py 中只有 skill_calculator，time_skill.py 中只有 skill_time
    # 这保证非 skill_ 前缀的全局变量不会污染
    n = autoload_skills(sm, ["voice_agent.agent.skills.calculator_skill"])
    assert n == 1
    assert "calculate" in sm.get_skill_names()