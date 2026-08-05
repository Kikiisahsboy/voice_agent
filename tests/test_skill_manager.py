"""SkillManager 与 Skill 注册 / 注入测试。"""

from voice_agent.agent.skill_manager import Skill, SkillManager


def make_skill(name, handler):
    return Skill(
        name=name,
        description=f"test {name}",
        parameters={"type": "object", "properties": {}},
        handler=handler,
        category="test",
    )


def test_register_and_list():
    sm = SkillManager()
    s = make_skill("ping", lambda: "pong")
    sm.register(s)
    assert "ping" in sm.get_skill_names()
    assert sm.get_skill_names() == ["ping"]


def test_execute_simple():
    sm = SkillManager()
    sm.register(make_skill("add", lambda a, b: a + b))
    assert sm.execute("add", {"a": 2, "b": 3}) == "5"


def test_execute_string_args_json():
    sm = SkillManager()
    sm.register(make_skill("add", lambda a, b: a + b))
    assert sm.execute("add", '{"a": 1, "b": 2}') == "3"


def test_execute_unknown_skill():
    sm = SkillManager()
    assert "未找到" in sm.execute("missing", {})


def test_execute_with_context_injection():
    """handler 接受 _context 参数，应由 SkillManager 注入。"""
    received = {}

    def handler(x, _context=None):
        received["x"] = x
        received["ctx"] = _context
        return "ok"

    sm = SkillManager()
    sm.register(make_skill("with_ctx", handler))
    sm.execute("with_ctx", {"x": 1}, context={"memory": "fake"})
    assert received["x"] == 1
    assert received["ctx"] == {"memory": "fake"}


def test_execute_missing_required_param():
    sm = SkillManager()

    def handler(a):
        return a

    sm.register(make_skill("req", handler))
    result = sm.execute("req", {})
    assert "缺少必填参数" in result


def test_execute_handler_no_params():
    sm = SkillManager()
    sm.register(make_skill("greet", lambda: "hello"))
    assert sm.execute("greet", {}) == "hello"
    assert sm.execute("greet", {"unexpected": "x"}) == "hello"


def test_get_all_tools_ollama_format():
    sm = SkillManager()
    sm.register(make_skill("foo", lambda: ""))
    tools = sm.get_all_tools()
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "foo"


def test_register_overwrite():
    sm = SkillManager()
    sm.register(make_skill("x", lambda: "v1"))
    sm.register(make_skill("x", lambda: "v2"))
    assert sm.execute("x", {}) == "v2"