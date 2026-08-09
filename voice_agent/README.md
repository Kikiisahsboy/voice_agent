# 🎙️ Voice Agent

> 基于 Plan-and-Execute + ReAct + Reflection 的端到端语音对话 Agent，支持双层记忆、文档型 RAG、MCP 协议扩展与流式 TTS 推送。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io)

---

## ✨ 核心特性

- 🧠 **Plan-and-Execute + ReAct + Reflection** 三层 Agent 范式，自动编排工具调用
- 💾 **双层记忆 + 三层分类**：滑动窗口短期记忆 + ChromaDB 长期记忆（fact / preference / event）+ 用户画像
- 🎯 **原生 metadata 过滤**：`memory_types` / `time_range` 直接作用于 ChromaDB where 子句，避免应用层 post-filter 的过度召回
- 📚 **文档型 RAG**：Hybrid 检索（向量 + BM25 + RRF + Rerank）+ 段落/句子/滑窗分块
- 🔌 **MCP 双端**：Client 长连接接入外部工具，Server 反向暴露自身能力
- 🌊 **流式端到端**：LLM token + edge-tts 音频 base64 SSE 实时推送，边生成边播报
- 🛑 **Barge-in 打断**：request_id 级 Event 标志支持用户随时中断
- 🧩 **Skill 插件化**：importlib 反射 autoload + inspect 签名依赖注入，新增 Skill 零侵入
- 📊 **结构化日志**：trace_id 全链路追踪 + JSON 友好输出

---

## 🏗️ 架构图

```
                        ┌──────────────────────────────────────────┐
                        │            Web Frontend (HTML/JS)         │
                        │  · SSE 客户端  · Web Audio API 播放音频    │
                        └────────────────┬─────────────────────────┘
                                         │  SSE (text_chunk / audio_chunk / tool_call)
                                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          Flask Web Server (server.py)                       │
│  · /api/chat/stream        · /api/chat/voice/stream                       │
│  · /api/rag/load|search    · /api/memory/*|profile|delete                 │
│  · /api/skills             · /api/health   · /api/abort (barge-in)        │
└────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                       Agent Orchestrator (orchestrator.py)                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  Plan    │ →  │  ReAct   │ →  │   Tool Call  │ →  │  Reflection  │ →  │
│  │ (列计划) │    │(推理-调用)│   │(带重试+修复) │    │ (查漏补缺)   │    │
│  └──────────┘    └──────────┘    └──────┬───────┘    └──────────────┘    │
│                                          ▼                                  │
│                              ┌─────────────────────┐                      │
│                              │ SkillManager (autoload)                   │
│                              │  · get_current_time  · calculator          │
│                              │  · search_memory    · search_knowledge    │
│                              │  · get_weather      · web_search          │
│                              └──────────┬──────────┘                      │
└─────────────────────────────────────────┼──────────────────────────────────┘
                                          │
        ┌───────────────┬─────────────────┼─────────────────┬───────────────┐
        ▼               ▼                 ▼                 ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Ollama LLM  │ │ Memory       │ │ Document RAG │ │   MCP        │ │   ASR/TTS    │
│  (qwen3:8b)  │ │ Manager      │ │ Hybrid       │ │  Client+     │ │  Vosk+       │
│ Function     │ │              │ │ Retriever    │ │  Server      │ │  edge-tts    │
│ Calling      │ │ · 短期窗口   │ │ · 向量+BM25  │ │ · stdio 长连 │ │ · 流式推送    │
│              │ │ · 三层分类   │ │ · RRF 融合   │ │ · 协议互操作 │ │ · 浏览器播放 │
│              │ │ · 用户画像   │ │ · 段落分块   │ │              │ │ · 打断       │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

---

## 📁 目录结构

```
voice_agent/
├── main.py                  # 入口：加载配置 + 装配服务 + 启动 Web
├── config.yaml              # 主配置（LLM/ASR/TTS/Memory/Skills/MCP）
├── environment.yml          # Conda 环境
├── start.bat                # Windows 一键启动
│
├── llm/                     # LLM 客户端
│   └── stream_ollama_client.py     # Ollama 流式 + Function Calling
│
├── asr/                     # 语音识别
│   └── stream_asr.py               # Vosk 流式识别
│
├── tts/                     # 语音合成
│   └── stream_tts.py               # edge-tts + 浏览器推送
│
├── agent/                   # Agent 核心
│   ├── orchestrator.py             # Plan + ReAct + Reflection 主循环
│   ├── memory_manager.py           # 双层记忆 + 画像 + 抽取
│   ├── skill_manager.py            # Skill 注册 + 路由 + 依赖注入
│   ├── skills_autoloader.py        # 反射扫描 skill_* 全局对象
│   └── skills/                     # 内置技能（autoload 自动发现）
│       ├── time_skill.py           #   · get_current_time
│       ├── calculator_skill.py     #   · calculate
│       ├── memory_skill.py         #   · search_memory
│       ├── knowledge_skill.py      #   · search_knowledge（RAG）
│       ├── weather_skill.py        #   · get_weather（wttr.in）
│       └── web_search_skill.py     #   · web_search（DuckDuckGo）
│
├── memory/                  # 记忆与 RAG
│   ├── chroma_store.py             # ChromaDB 封装 + 衰减
│   └── hybrid_retriever.py         # BM25 + 向量 + RRF + DocumentChunker
│
├── mcp/                     # MCP 协议
│   ├── mcp_client.py               # 长连接 Client（独立 event loop）
│   └── mcp_server.py               # 反向暴露 Agent 能力
│
├── web/                     # Web 服务
│   ├── server.py                   # Flask + SSE + REST API
│   └── static/index.html           # 聊天 UI（音频流式播放）
│
├── chroma_data/             # ChromaDB 对话记忆持久化（git 忽略）
└── chroma_data_docs/        # ChromaDB 文档库持久化（git 忽略）
```

---

## 🚀 快速开始

### 1. 环境要求

- **Python** 3.10+
- **Ollama** 已安装并运行（[下载地址](https://ollama.com/download)）
- **Vosk** 中文模型（[下载地址](https://alphacephei.com/vosk/models)），解压后配置路径
- **ffmpeg**（可选，TTS 解码需要；Windows 通常自带）
- **麦克风**（语音输入需要）

### 2. 拉取 LLM 模型

```bash
ollama pull qwen3:8b     # 或 qwen2.5:7b 等支持 Function Calling 的模型
```

### 3. 安装依赖

```bash
# 推荐：使用 conda
conda env create -f environment.yml
conda activate voice_agent

# 或使用 pip
pip install flask flask-cors requests numpy chromadb sentence-transformers \
            vosk edge-tts miniaudio pyaudio pyyaml mcp
```

### 4. 配置

编辑 `config.yaml`，重点关注：

```yaml
llm:
  ollama:
    base_url: "http://localhost:11434"
    model: "qwen3:8b"          # 替换为你拉取的模型

asr:
  vosk:
    model_path: "D:/vosk-model/vosk-model-cn-kaldi-multicn-0.15"
    # ↑ 替换为你的 Vosk 模型实际路径
```

### 5. 启动

```bash
# 方式一：直接运行
python -m voice_agent.main

# 方式二：Windows 用户
start.bat
```

打开浏览器访问 **http://localhost:8081**，即可开始对话。

---

## 🎯 使用示例

### 文本对话
> 用户：北京今天适合户外跑步吗？
>
> Agent：让我查一下…（plan → react → reflect）
> 北京今天晴 25°C，湿度 50%，很适合跑步。如果慢跑 1 小时，大约消耗 500 大卡，建议早餐吃高蛋白食物补充…（流式返回 + 实时合成语音）

### 加载知识库
```bash
curl -X POST http://localhost:8081/api/rag/load \
  -H "Content-Type: application/json" \
  -d '{"file_path": "C:/docs/产品手册.txt"}'
```

### 打断 Agent（Barge-in）
```bash
curl -X POST http://localhost:8081/api/abort \
  -H "Content-Type: application/json" \
  -d '{"request_id": "req_xxx"}'
```

### 启用 MCP
```yaml
# config.yaml
mcp:
  client:
    enabled: true
    servers:
      - name: "my_tool_server"
        command: "python"
        args: ["-m", "my_mcp_server"]
```

---

## 📡 API 文档

### REST API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/health` | GET | 健康检查（含 RAG chunk 数、Skill 列表）|
| `/api/skills` | GET | 列出所有已加载 Skill |
| `/api/chat/stream` | POST (SSE) | 流式文本对话 + TTS 音频 |
| `/api/chat/voice/stream` | POST (SSE) | 流式语音对话（上传 WAV）|
| `/api/abort` | POST | 中断进行中的请求（Barge-in）|
| `/api/rag/load` | POST | 加载文档到 RAG（file_path 或 text）|
| `/api/rag/search` | POST | 手动查询 RAG |
| `/api/rag/stats` | GET | RAG 统计信息 |
| `/api/memory/recent` | GET | 获取最近 N 条长期记忆 |
| `/api/memory/list` | GET | 列出全部长期记忆 |
| `/api/memory/profile` | GET | 获取用户画像 |
| `/api/memory/delete` | POST | 按 id 或 keyword 删除记忆 |
| `/api/reset` | POST | 重置当前会话短期记忆 |

### SSE 事件类型（`/api/chat/stream`）

| 类型 | 说明 |
|---|---|
| `status` | 状态切换（thinking / listening / processing）|
| `plan` | 生成的计划文本 |
| `llm_chunk` | LLM 输出 token（流式）|
| `sentence` | 完整句子（用于 TTS 触发）|
| `tool_call` | 工具调用请求 |
| `tool_result` | 工具调用结果 |
| `reflection` | 反思结果（complete / incomplete）|
| `audio_chunk` | TTS 合成音频（base64 mp3）|
| `done` | 本轮对话结束 |
| `error` | 错误信息 |
| `aborted` | 已被客户端打断 |

---

## 🧩 开发自定义 Skill

只需新建 `voice_agent/agent/skills/my_skill.py`：

```python
from voice_agent.agent.skill_manager import Skill

def _my_handler(arg1: str, _context=None) -> str:
    # 业务逻辑（可选：使用 _context["memory"] / _context["document_rag"]）
    return f"处理 {arg1} 的结果"

skill_my_thing = Skill(
    name="my_thing",
    description="何时使用（让 LLM 知道触发场景）",
    parameters={
        "type": "object",
        "properties": {
            "arg1": {"type": "string", "description": "参数说明"}
        },
        "required": ["arg1"]
    },
    handler=_my_handler,
    category="custom",
)
```

并在 `config.yaml` 中加入：

```yaml
skills:
  autoload:
    - "voice_agent.agent.skills.my_skill"
```

**无需改 main.py**——`autoload_skills` 会自动扫描模块中所有以 `skill_` 开头的全局变量。

---

## ⚙️ 配置详解

```yaml
llm:
  react:
    max_rounds: 5                  # ReAct 最大推理-调用轮数
    enable_planning: true          # 是否启用 Plan 阶段
    enable_reflection: true        # 是否启用 Reflection 阶段
    tool_retry_limit: 2            # 工具调用重试次数

memory:
  short_term:
    max_turns: 10                  # 滑动窗口保留轮数
  long_term:
    enabled: true
    chroma:
      persist_dir: "./voice_agent/chroma_data"
      collection_name: "conversation_memory_v2"
    embedding:
      model: "BAAI/bge-small-zh-v1.5"     # 中文 Embedding
    retrieval:
      top_k: 3
      similarity_threshold: 0.5
    decay:
      enabled: true
      max_age_days: 30
      max_total_memories: 1000

conversation:
  system_prompt: |
    你是一个贴心的智能语音助手。
    # LLM 的人格与行为约束
```

---

## 🧪 Roadmap

### 已实现 ✅
- [x] Plan-and-Execute + ReAct + Reflection 三层 Agent
- [x] 双层记忆 + fact/preference/event 三层分类
- [x] 用户画像持续抽取
- [x] 原生 metadata 过滤（build_where 自动转 ChromaDB where 子句）
- [x] 文档型 RAG（Hybrid Retriever + DocumentChunker）
- [x] MCP Client 长连接 + Server 反向暴露
- [x] 流式 LLM token + edge-tts 音频 SSE 推送
- [x] Barge-in 用户打断
- [x] Skill autoload + 依赖注入
- [x] 结构化日志（trace_id）

### 计划中 🚧
- [ ] Cross-encoder Rerank（bge-reranker-base）替代简易打分
- [ ] Query Rewrite + HyDE 提升 RAG 召回
- [ ] 记忆冲突检测（Mem0 风格 ADD/UPDATE/DELETE）
- [ ] 用户画像独立持久化（当前仅内存）
- [ ] 工具调用并行化（独立 tool 并行 gather）
- [ ] PDF / Word / Markdown 文档加载
- [ ] WebSocket 升级（替代 SSE，支持双向流）
- [ ] 自动化评估集（10+ golden cases 回归）
- [ ] Docker 一键部署

---

## 🐛 已知问题

1. **Ollama Function Calling 流式拼接**：极小概率 tool_calls 参数跨 chunk 不完整（升级 Ollama 可缓解）
2. **edge-tts 网络依赖**：离线环境不可用，可换 Coqui TTS / pyttsx3
3. **当前 Rerank 是关键词重叠率**：精度有限，建议接入 bge-reranker
4. **MCP Server 未鉴权**：生产环境需加认证

---

## 🤝 贡献

欢迎 PR / Issue！

特别欢迎的方向：
- 新 Skill 接入（地图 / 日历 / 邮件 / 翻译…）
- RAG 召回策略优化
- 评估集 & Benchmark 建设
- 多语言支持

---

## 📄 License

MIT License. 详见 [LICENSE](LICENSE)。

---

## 🙏 致谢

- [Ollama](https://ollama.com) — 本地 LLM 推理
- [Vosk](https://alphacephei.com/vosk) — 离线 ASR
- [edge-tts](https://github.com/rany2/edge-tts) — 流式 TTS
- [ChromaDB](https://www.trychroma.com) — 向量数据库
- [BAAI/bge](https://huggingface.co/BAAI) — 中文 Embedding
- [Model Context Protocol](https://modelcontextprotocol.io) — 工具协议标准