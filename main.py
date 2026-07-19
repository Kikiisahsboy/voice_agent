# -*- coding: utf-8 -*-
"""Voice Agent 入口 — 初始化所有服务并启动 Web 服务器。"""

import argparse
import logging
import os
import sys
import threading

import yaml

# 确保项目根目录在路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _setup_logging(cfg: dict):
    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get(
            "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Voice Agent")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--port", type=int, default=None, help="Web 端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    _setup_logging(cfg)

    logger = logging.getLogger("voice_agent")
    logger.info("Voice Agent 启动中...")

    # ── 1. LLM 客户端 ──────────────────────────
    from voice_agent.llm.stream_ollama_client import StreamOllamaClient

    llm_cfg = cfg["llm"]["ollama"]
    llm_client = StreamOllamaClient(
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 0.7),
        num_predict=llm_cfg.get("num_predict", 2048),
        timeout=llm_cfg.get("timeout", 120.0),
    )

    if not llm_client.health_check():
        logger.error(
            "Ollama 服务不可用或模型 '%s' 未找到，请先启动 Ollama 并拉取模型。",
            llm_cfg["model"],
        )
        sys.exit(1)

    llm_client.warm_up()

    # ── 2. ASR 服务 ────────────────────────────
    from voice_agent.asr.stream_asr import StreamASR

    asr_cfg = cfg["asr"]["vosk"]
    asr_service = None
    if os.path.isdir(asr_cfg["model_path"]):
        asr_service = StreamASR(
            model_path=asr_cfg["model_path"],
            sample_rate=asr_cfg.get("sample_rate", 16000),
        )
    else:
        logger.warning(
            "Vosk 模型路径不存在: %s，语音识别不可用", asr_cfg["model_path"]
        )

    # ── 3. TTS 服务 ────────────────────────────
    from voice_agent.tts.stream_tts import StreamTTS

    tts_cfg = cfg["tts"]["edge_tts"]
    try:
        tts_service = StreamTTS(sample_rate=tts_cfg.get("sample_rate", 24000))
    except ImportError as e:
        logger.warning("TTS 不可用: %s", e)
        tts_service = None

    # ── 4. 记忆系统 ───────────────────────────
    from voice_agent.agent.memory_manager import MemoryManager
    from voice_agent.memory.chroma_store import ChromaMemoryStore

    mem_cfg = cfg["memory"]
    chroma_store = ChromaMemoryStore(
        persist_dir=mem_cfg["long_term"]["chroma"]["persist_dir"],
        collection_name=mem_cfg["long_term"]["chroma"]["collection_name"],
    )
    memory_manager = MemoryManager(
        chroma_store=chroma_store,
        short_term_max_turns=mem_cfg["short_term"]["max_turns"],
        long_term_enabled=mem_cfg["long_term"]["enabled"],
    )
    memory_manager.set_system_prompt(cfg["conversation"]["system_prompt"])

    # ── 5. Skill 系统 ──────────────────────────
    from voice_agent.agent.skill_manager import SkillManager

    skill_manager = SkillManager()

    # 注册内置 skills
    from voice_agent.agent.skills.time_skill import skill_time
    from voice_agent.agent.skills.calculator_skill import skill_calculator
    from voice_agent.agent.skills.memory_skill import skill_memory_search

    skill_manager.register(skill_time)
    skill_manager.register(skill_calculator)
    skill_manager.register(skill_memory_search)

    # ── 6. MCP Client（可选）───────────────────
    mcp_client = None
    if cfg["mcp"]["client"]["enabled"]:
        from voice_agent.mcp.mcp_client import MCPClient

        mcp_client = MCPClient()
        import asyncio

        for server_cfg in cfg["mcp"]["client"]["servers"]:
            asyncio.run(
                mcp_client.connect(
                    server_command=server_cfg["command"],
                    server_args=server_cfg.get("args", []),
                    server_name=server_cfg.get("name", "default"),
                )
            )

    # ── 7. Agent 编排器 ────────────────────────
    from voice_agent.agent.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        asr_service=asr_service,
        tts_service=tts_service,
        memory_manager=memory_manager,
        skill_manager=skill_manager,
        mcp_client=mcp_client,
        max_react_rounds=cfg["llm"]["react"]["max_rounds"],
    )

    # ── 8. MCP Server（可选，线程启动）─────────
    if cfg["mcp"]["server"]["enabled"]:
        from voice_agent.mcp.mcp_server import AgentMCPServer

        mcp_server = AgentMCPServer(
            memory_manager=memory_manager,
            skill_manager=skill_manager,
        )

        def _run_mcp():
            asyncio.run(mcp_server.start())

        threading.Thread(target=_run_mcp, daemon=True).start()
        logger.info("MCP Server 已启动（后台线程）")

    # ── 9. Web 服务 ────────────────────────────
    from voice_agent.web.server import app, init_services, run_server

    init_services(orchestrator, asr_service, tts_service, llm_client)

    port = args.port or cfg["server"]["port"]
    debug = args.debug or cfg["server"]["debug"]

    import asyncio

    print(f"\n{'='*50}")
    print(f"  Voice Agent 已就绪")
    print(f"  Web 界面: http://localhost:{port}")
    print(f"  LLM 模型: {llm_cfg['model']}")
    print(f"  ASR: {'可用' if asr_service else '不可用'}")
    print(f"  TTS: {'可用' if tts_service else '不可用'}")
    print(f"  Skills: {skill_manager.get_skill_names()}")
    print(f"  MCP Client: {'已启用' if mcp_client else '未启用'}")
    print(f"{'='*50}\n")

    run_server(host=cfg["server"]["host"], port=port, debug=debug)


if __name__ == "__main__":
    main()
