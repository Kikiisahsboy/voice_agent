# -*- coding: utf-8 -*-
"""Flask Web 服务 — 为语音 Agent 提供 SSE 流式 API。"""

import io
import json
import logging
import os
import sys
import wave
from typing import Optional

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# 全局服务实例（由 main.py 注入）
_orchestrator = None  # AgentOrchestrator
_asr = None  # StreamASR
_tts = None  # StreamTTS
_llm = None  # StreamOllamaClient


def init_services(orchestrator, asr_service, tts_service, llm_client):
    global _orchestrator, _asr, _tts, _llm
    _orchestrator = orchestrator
    _asr = asr_service
    _tts = tts_service
    _llm = llm_client
    logger.info("Web 服务已绑定 Agent 实例")


# ── SSE 辅助 ──────────────────────────────────────────

def _sse_event(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


# ── API 路由 ──────────────────────────────────────────

@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """流式文本聊天 SSE 端点。"""
    data = request.get_json(force=True)
    user_text = data.get("text", "").strip()
    if not user_text:
        return jsonify({"error": "text is required"}), 400

    def generate():
        for event in _orchestrator.process_text_input(user_text):
            yield _sse_event(event)
    return Response(
        generate(),
        mimetype="text/plain",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.route("/api/chat/voice/stream", methods=["POST"])
def chat_voice_stream():
    """流式语音聊天 SSE 端点。"""
    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    hex_preview = audio_bytes[:20].hex(' ') if len(audio_bytes) >= 20 else audio_bytes.hex(' ')
    text_preview = ''.join(chr(b) if 32 <= b < 127 else '.' for b in audio_bytes[:20])
    logger.info("收到音频: 大小=%d 字节, 前20字节hex=[%s], 文本=[%s]", len(audio_bytes), hex_preview, text_preview)

    # 将 WAV 转为 int16 PCM 块列表
    try:
        audio_chunks = _wav_to_pcm_chunks(audio_bytes)
        if not audio_chunks:
            return jsonify({"error": "音频解析失败"}), 400
    except Exception as e:
        logger.error("音频解析错误: %s", e)
        return jsonify({"error": str(e)}), 400

    def generate():
        for event in _orchestrator.process_voice_input(audio_chunks):
            yield _sse_event(event)
    return Response(
        generate(),
        mimetype="text/plain",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.route("/api/chat/text", methods=["POST"])
def chat_text():
    """非流式文本聊天，返回完整结果 + TTS 音频（兼容旧接口）。"""
    import asyncio
    import base64

    data = request.get_json(force=True)
    user_text = data.get("text", "").strip()
    if not user_text:
        return jsonify({"error": "text is required"}), 400

    full_text = ""
    for event in _orchestrator.process_text_input(user_text):
        if event["type"] == "text_chunk":
            full_text += event.get("content", "")
        elif event["type"] == "done":
            full_text = event.get("content", full_text)
        elif event["type"] == "error":
            full_text = event.get("content", full_text)

    # TTS → base64
    audio_b64 = ""
    if full_text:
        try:
            import edge_tts
            chunks = []

            async def _run():
                communicate = edge_tts.Communicate(full_text, "zh-CN-XiaoxiaoNeural")
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk["data"])

            asyncio.run(_run())
            if chunks:
                audio_b64 = base64.b64encode(b"".join(chunks)).decode("utf-8")
        except Exception as e:
            logger.warning("TTS 合成失败: %s", e)

    return jsonify({
        "text": full_text,
        "audio": audio_b64,
        "audio_mime": "audio/mp3",
    })


@app.route("/api/reset", methods=["POST"])
def reset_conversation():
    """重置对话历史。"""
    if _orchestrator:
        _orchestrator.reset_conversation()
    return jsonify({"ok": True})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "asr": _asr is not None,
        "llm": _llm is not None,
        "tts": _tts is not None,
    })


@app.route("/api/skills", methods=["GET"])
def list_skills():
    """列出所有可用技能。"""
    if _orchestrator and _orchestrator._skills:
        return jsonify({
            "skills": _orchestrator._skills.get_skill_names(),
        })
    return jsonify({"skills": []})


@app.route("/")
def index():
    response = send_from_directory(app.static_folder, "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# ── 辅助函数 ──────────────────────────────────────────

def _wav_to_pcm_chunks(wav_bytes: bytes, chunk_size: int = 4096) -> list[bytes]:
    """将 WAV 字节数据转为 int16 PCM 块列表。"""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sw == 2:
        audio = np.frombuffer(frames, dtype=np.int16)
    elif sw == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128) * 256
    else:
        return []

    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1).astype(np.int16)

    raw = audio.tobytes()
    chunks = [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)]
    return chunks


# ── 入口 ──────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8081, debug: bool = False):
    logger.info("Voice Agent Web 服务启动: http://%s:%s", host, port)
    app.run(host=host, port=port, debug=debug)
