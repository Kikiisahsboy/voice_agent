"""内置 skills 单元测试。"""

from voice_agent.agent.skills.calculator_skill import skill_calculator
from voice_agent.agent.skills.time_skill import skill_time


def test_calculator_basic():
    assert skill_calculator.handler("1+2") == "3"
    assert skill_calculator.handler("10*5") == "50"
    # 100/4 = 25.0，但内置会规整为整数 25
    assert skill_calculator.handler("100/4") == "25"
    assert skill_calculator.handler("2**8") == "256"


def test_calculator_float_to_int():
    assert skill_calculator.handler("10/2") == "5"


def test_calculator_safety():
    """不应支持函数调用、属性访问。"""
    out = skill_calculator.handler("__import__('os').system('echo pwned')")
    assert "出错" in out or "不支持" in out


def test_calculator_error_handling():
    out = skill_calculator.handler("1/0")
    assert "出错" in out


def test_time_default_format():
    out = skill_time.handler()
    assert "年" in out
    assert ":" in out


def test_time_date_only():
    out = skill_time.handler(format="date")
    assert "年" in out
    assert ":" not in out


def test_time_time_only():
    out = skill_time.handler(format="time")
    assert ":" in out
    assert "年" not in out