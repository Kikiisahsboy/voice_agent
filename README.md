# Voice Agent

基于 ReAct 推理框架的智能语音助手，支持语音识别（ASR）→ 大语言模型（LLM）→ 语音合成（TTS）全链路流式对话，集成双层记忆系统与可扩展技能框架。

## 架构概览

```
用户语音/文本输入
       │
       ▼
  ┌──────────┐     ┌──────────────────┐     ┌──────────┐
  │  ASR     │────▶│  Agent 编排器     │────▶│  TTS     │
  │ (Vosk)   │     │  (ReAct 循环)     │     │ (Edge)   │
  └──────────┘     │                  │     └──────────┘
                   │  ┌────────────┐  │
                   │  │ LLM Client │  │
                   │  │ (Ollama)   │  │
                   │  └────────────┘  │
                   │                  │
                   │  ┌────────────┐  │
                   │  │ 记忆系统    │  │
                   │  │ 短期+长期   │  │
                   │  └────────────┘  │
                   │                  │
                   │  ┌────────────┐  │
                   │  │ 技能系统    │  │
                   │  │ Skill+MCP  │  │
                   │  └────────────┘  │
                   └──────────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Flask Web   │
                    │  SSE 流式 API │
                    └──────────────┘
```

## 核心特性

- **流式全链路** — ASR 识别、LLM 生成、TTS 合成均支持流式输出，首字延迟极低
- **ReAct 推理** — Agent 编排器执行多轮"思考→调用工具→观察结果"循环，支持复杂任务拆解
- **双层记忆** — 短期记忆（滑动窗口）+ 长期记忆（ChromaDB 向量检索 + 自动摘要）
- **技能系统** — 可插拔的 Skill 注册机制，内置时间查询、计算器、记忆搜索
- **MCP 协议** — 支持 MCP Client 连接外部工具服务器，支持 MCP Server 暴露 Agent 能力
- **Web 界面** — Flask 提供 SSE 流式 API 和前端页面，支持文本/语音双模交互

## 项目结构

```
voice_agent-master/
├── main.py                  # 入口，初始化所有服务并启动 Web
├── config.yaml              # 全局配置文件
├── environment.yml          # Conda 环境依赖
├── start.bat                # Windows 一键启动脚本
├── agent/                   # Agent 核心
│   ├── orchestrator.py      # ReAct 编排器，协调 ASR→Memory→LLM→TTS
│   ├── skill_manager.py     # 技能注册中心
│   ├── memory_manager.py    # 双层记忆管理器
│   └── skills/              # 内置技能
│       ├── time_skill.py    # 时间查询
│       ├── calculator_skill.py  # 数学计算
│       └── memory_skill.py  # 记忆搜索
├── asr/
│   └── stream_asr.py        # Vosk 流式语音识别
├── llm/
│   └── stream_ollama_client.py  # Ollama 流式客户端（含 Tool Calling）
├── tts/
│   └── stream_tts.py        # Edge-TTS 流式语音合成
├── memory/
│   └── chroma_store.py      # ChromaDB 长期记忆存储（向量检索+衰减）
├── mcp/
│   ├── mcp_client.py        # MCP 客户端，连接外部 MCP Server
│   └── mcp_server.py        # MCP 服务端，暴露 Agent 能力
├── web/
│   ├── server.py            # Flask Web 服务 + SSE 流式 API
│   └── static/              # 前端静态文件
│       └── index.html
└── voice_agent/
    └── chroma_data/         # ChromaDB 持久化数据目录
```

## 快速开始

### 1. 环境准备

**安装 Conda 环境：**

```bash
conda env create -f environment.yml
conda activate voice_agent
```

**安装 Ollama 并拉取模型：**

```bash
# 安装 Ollama: https://ollama.com/download
ollama pull qwen3:8b
ollama serve
```

**下载 Vosk 中文模型：**

从 [Vosk Models](https://alphacephei.com/vosk/models) 下载 `vosk-model-cn-kaldi-multicn-0.15`，解压到本地路径（如 `D:/vosk-model/vosk-model-cn-kaldi-multicn-0.15`）。

### 2. 修改配置

编辑 `config.yaml`，确认以下关键配置：

```yaml
asr:
  vosk:
    model_path: "你的Vosk模型路径"    # 修改为实际路径

llm:
  ollama:
    base_url: "http://localhost:11434"
    model: "qwen3:8b"                 # 修改为已拉取的模型
```

### 3. 启动服务

**方式一：Windows 一键启动**

```bash
start.bat
```

**方式二：命令行启动**

```bash
conda activate voice_agent
python main.py --config config.yaml
```

启动后访问 `http://localhost:8081` 即可使用 Web 界面。

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/chat/stream` | POST | 流式文本聊天（SSE） |
| `/api/chat/voice/stream` | POST | 流式语音聊天（SSE，上传 WAV 音频） |
| `/api/chat/text` | POST | 非流式文本聊天（返回 JSON + TTS 音频） |
| `/api/reset` | POST | 重置对话历史 |
| `/api/health` | GET | 健康检查 |
| `/api/skills` | GET | 列出可用技能 |

## 配置说明

配置文件为 `config.yaml`，主要配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `server.host/port` | Web 服务地址 | `0.0.0.0:8081` |
| `asr.vosk.model_path` | Vosk 模型路径 | — |
| `llm.ollama.model` | Ollama 模型名称 | `qwen3:8b` |
| `llm.react.max_rounds` | ReAct 最大推理轮次 | `5` |
| `tts.edge_tts.voice` | TTS 语音 | `zh-CN-XiaoxiaoNeural` |
| `memory.short_term.max_turns` | 短期记忆轮数 | `10` |
| `memory.long_term.enabled` | 是否启用长期记忆 | `true` |
| `skills.autoload` | 自动加载的技能列表 | 时间/计算器/记忆 |
| `mcp.client.enabled` | 是否启用 MCP Client | `false` |
| `mcp.server.enabled` | 是否启用 MCP Server | `false` |

## 技能扩展

在 `agent/skills/` 下新建文件，定义一个 `Skill` 对象并注册即可：

```python
from voice_agent.agent.skill_manager import Skill

def _my_handler(param: str) -> str:
    return f"处理结果: {param}"

skill_my = Skill(
    name="my_skill",
    description="我的自定义技能描述",
    parameters={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "参数说明"}
        },
        "required": ["param"],
    },
    handler=_my_handler,
    category="custom",
)
```

然后在 `main.py` 中导入并注册：

```python
from voice_agent.agent.skills.my_skill import skill_my
skill_manager.register(skill_my)
```

## 依赖

- **Python** 3.11+
- **Flask** — Web 框架
- **Vosk** — 离线语音识别
- **Ollama** — 本地大语言模型推理
- **Edge-TTS** — 微软语音合成
- **ChromaDB** — 向量数据库（长期记忆）
- **Sentence-Transformers** — 文本向量化（BAAI/bge-small-zh-v1.5）
- **MCP SDK** — Model Context Protocol 支持
- **PyAudio** — 音频播放
- **WebRTC VAD** — 语音活动检测
