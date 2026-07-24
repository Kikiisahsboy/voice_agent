"""MCP Client 单测：start/stop、connect 失败时不抛、call_tool 未连接返回错误。

真实 MCP Server 连接需要外部依赖，这里只验证不抛异常 + 未连接时安全降级。
"""

from voice_agent.mcp.mcp_client import MCPClient


def test_init_safely():
    client = MCPClient()
    assert client._tools == []
    assert client._sessions == {}


def test_call_tool_without_connect_returns_error():
    client = MCPClient()
    out = client.call_tool("any_tool", {})
    assert "未找到" in out or "未连接" in out


def test_connect_failure_does_not_raise():
    """连接失败时应返回 False 而不是抛异常。"""
    client = MCPClient()
    ok = client.connect(
        server_command="definitely_not_a_real_binary_xyz",
        server_args=[],
        server_name="bogus",
    )
    assert ok is False


def test_get_tools_empty_before_connect():
    client = MCPClient()
    assert client.get_tools_as_ollama_format() == []


def test_get_tools_ollama_format_with_mock_data():
    """手动注入工具，验证 Ollama 格式转换。"""
    client = MCPClient()
    client._tools.append({
        "name": "t1",
        "description": "test tool",
        "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
    })
    out = client.get_tools_as_ollama_format()
    assert len(out) == 1
    assert out[0]["function"]["name"] == "t1"
    assert out[0]["function"]["parameters"]["properties"]["x"]["type"] == "integer"


def test_get_tools_ollama_format_wraps_params():
    """如果 parameters 缺 type，应自动包成 object。"""
    client = MCPClient()
    client._tools.append({
        "name": "t2",
        "description": "no type",
        "parameters": {"properties": {}},
    })
    out = client.get_tools_as_ollama_format()
    assert out[0]["function"]["parameters"]["type"] == "object"