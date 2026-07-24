# -*- coding: utf-8 -*-
"""Flask Web 服务 — 为语音 Agent 提供 SSE 流式 API + 文档型 RAG 端点 + TTS 流推送。"""

import io
import json
import logging
import os
import queue
import threading
import time
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

# 全局服务实例
_orchestrator = None
_asr = None
_tts = None
_llm = None
_memory_manager = None
_document_rag = None

# 每个请求的 TTS 中断标志（用于 barge-in）
_abort_flags: dict[str, threading.Event] = {}
_abort_lock = threading.Lock()


def init_services(
    orchestrator,
    asr_service,
    tts_service,
    llm_client,
    memory_manager=None,
    document_rag=None,
):
    global _orchestrator, _asr, _tts, _llm, _memory_manager, _document_rag
    _orchestrator = orchestrator
    _asr = asr_service
    _tts = tts_service
    _llm = llm_client
    _memory_manager = memory_manager
    _document_rag = document_rag
    logger.info("Web 服务已绑定 Agent 实例 (RAG: %s)",
                "yes" if document_rag else "no")


def _sse_event(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


def _register_abort(request_id: str) -> threading.Event:
    ev = threading.Event()
    with _abort_lock:
        _abort_flags[request_id] = ev
    return ev


def _pop_abort(request_id: str) -> Optional[threading.Event]:
    with _abort_lock:
        return _abort_flags.pop(request_id, None)


# ── 路由 ────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "asr": _asr is not None,
        "llm": _llm is not None,
        "tts": _tts is not None,
        "rag_chunks": _document_rag.doc_count if _document_rag else 0,
        "skills": _orchestrator._skills.get_skill_names() if _orchestrator else [],
    })


@app.route("/api/skills", methods=["GET"])
def list_skills():
    if _orchestrator and _orchestrator._skills:
        return jsonify({"skills": _orchestrator._skills.get_skill_names()})
    return jsonify({"skills": []})


@app.route("/api/reset", methods=["POST"])
def reset_conversation():
    if _orchestrator:
        _orchestrator.reset_conversation()
    return jsonify({"ok": True})


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """流式文本聊天 SSE 端点 + TTS 音频推送。

    SSE 事件类型：
        status, plan, llm_chunk, sentence, tool_call, tool_result,
        reflection, audio_chunk, done, error, aborted
    """
    data = request.get_json(force=True)
    user_text = data.get("text", "").strip()
    request_id = data.get("request_id") or f"req_{int(time.time()*1000)}"
    enable_tts = bool(data.get("tts", True))

    if not user_text:
        return jsonify({"error": "text is required"}), 400

    abort_event = _register_abort(request_id)

    def generate():
        tts_queue: queue.Queue = queue.Queue(maxsize=8)
        tts_thread: Optional[threading.Thread] = None

        def _tts_worker():
            """后台线程：从队列取句子，合成 MP3，转 base64 推入 SSE 队列。"""
            while True:
                try:
                    item = tts_queue.get(timeout=0.5)
                except queue.Empty:
                    if abort_event.is_set():
                        return
                    continue
                if item is None:
                    return
                if abort_event.is_set():
                    return
                sentence = item
                if not sentence.strip() or _tts is None:
                    continue
                try:
                    b64 = _tts.synthesize_to_base64(sentence)
                    if b64 and not abort_event.is_set():
                        tts_out_queue.put({
                            "type": "audio_chunk",
                            "sentence": sentence,
                            "content": b64,
                            "mime": "audio/mp3",
                        })
                except Exception as e:
                    logger.warning("TTS 合成异常: %s", e)

        tts_out_queue: queue.Queue = queue.Queue(maxsize=64)
        if enable_tts and _tts is not None:
            tts_thread = threading.Thread(target=_tts_worker, daemon=True)
            tts_thread.start()

        # on_sentence 钩子：把句子塞给 TTS 工作线程
        def _on_sentence(sentence: str):
            if enable_tts and _tts is not None:
                try:
                    tts_queue.put_nowait(sentence)
                except queue.Full:
                    logger.debug("TTS 队列已满，丢弃句子")

        try:
            for event in _orchestrator.process_text_input(
                user_text,
                on_sentence=_on_sentence,
            ):
                if abort_event.is_set():
                    yield _sse_event({"type": "aborted"})
                    return

                yield _sse_event(event)

                # 抽空推一些 TTS 音频
                while True:
                    try:
                        audio_event = tts_out_queue.get_nowait()
                    except queue.Empty:
                        break
                    if abort_event.is_set():
                        yield _sse_event({"type": "aborted"})
                        return
                    yield _sse_event(audio_event)
        except GeneratorExit:
            logger.info("客户端断开: %s", request_id)
        finally:
            abort_event.set()
            try:
                tts_queue.put_nowait(None)
            except Exception:
                pass
            if tts_thread:
                tts_thread.join(timeout=1)
            _pop_abort(request_id)

    return Response(
        generate(),
        mimetype="text/plain",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.route("/api/chat/voice/stream", methods=["POST"])
def chat_voice_stream():
    """流式语音聊天。"""
    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    request_id = f"voice_{int(time.time()*1000)}"
    enable_tts = request.form.get("tts", "1") == "1"

    abort_event = _register_abort(request_id)

    try:
        audio_chunks = _wav_to_pcm_chunks(audio_bytes)
        if not audio_chunks:
            return jsonify({"error": "音频解析失败"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    def generate():
        tts_queue: queue.Queue = queue.Queue(maxsize=8)
        tts_thread = None

        def _tts_worker():
            while True:
                try:
                    item = tts_queue.get(timeout=0.5)
                except queue.Empty:
                    if abort_event.is_set():
                        return
                    continue
                if item is None or abort_event.is_set():
                    return
                try:
                    b64 = _tts.synthesize_to_base64(item) if _tts else ""
                    if b64 and not abort_event.is_set():
                        tts_out_queue.put({
                            "type": "audio_chunk",
                            "sentence": item,
                            "content": b64,
                            "mime": "audio/mp3",
                        })
                except Exception as e:
                    logger.warning("TTS 异常: %s", e)

        tts_out_queue: queue.Queue = queue.Queue(maxsize=64)
        if enable_tts and _tts is not None:
            tts_thread = threading.Thread(target=_tts_worker, daemon=True)
            tts_thread.start()

        def _on_sentence(sentence: str):
            if enable_tts and _tts is not None:
                try:
                    tts_queue.put_nowait(sentence)
                except queue.Full:
                    pass

        try:
            for event in _orchestrator.process_voice_input(
                audio_chunks,
                on_sentence=_on_sentence,
            ):
                if abort_event.is_set():
                    yield _sse_event({"type": "aborted"})
                    return
                yield _sse_event(event)

                while True:
                    try:
                        audio_event = tts_out_queue.get_nowait()
                    except queue.Empty:
                        break
                    if abort_event.is_set():
                        yield _sse_event({"type": "aborted"})
                        return
                    yield _sse_event(audio_event)
        except GeneratorExit:
            pass
        finally:
            abort_event.set()
            try:
                tts_queue.put_nowait(None)
            except Exception:
                pass
            if tts_thread:
                tts_thread.join(timeout=1)
            _pop_abort(request_id)

    return Response(
        generate(),
        mimetype="text/plain",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.route("/api/abort", methods=["POST"])
def abort_request():
    """中断一个进行中的请求（用于 barge-in）。"""
    data = request.get_json(force=True) or {}
    request_id = data.get("request_id", "")
    if not request_id:
        return jsonify({"error": "request_id required"}), 400
    with _abort_lock:
        ev = _abort_flags.get(request_id)
    if ev:
        ev.set()
        return jsonify({"ok": True})
    return jsonify({"ok": False, "msg": "no such request"}), 404


@app.route("/api/rag/load", methods=["POST"])
def rag_load():
    """加载文本到文档 RAG。"""
    if _document_rag is None:
        return jsonify({"error": "RAG 未启用"}), 400

    data = request.get_json(force=True) or {}
    if "file_path" in data:
        try:
            n = _document_rag.load_file(data["file_path"])
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    elif "text" in data:
        source = data.get("source", "inline")
        n = _document_rag.load_text(data["text"], source=source)
    else:
        return jsonify({"error": "file_path or text required"}), 400
    return jsonify({"ok": True, "chunks": n, "total": _document_rag.doc_count})


@app.route("/api/rag/search", methods=["POST"])
def rag_search():
    """手动查询 RAG。"""
    if _document_rag is None:
        return jsonify({"error": "RAG 未启用"}), 400
    data = request.get_json(force=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    top_k = int(data.get("top_k", 3))
    passages = _document_rag.search(query, top_k=top_k)
    return jsonify({"passages": passages, "count": len(passages)})


@app.route("/api/rag/stats", methods=["GET"])
def rag_stats():
    if _document_rag is None:
        return jsonify({"enabled": False, "chunks": 0})
    return jsonify({
        "enabled": True,
        "chunks": _document_rag.doc_count,
    })


@app.route("/api/memory/recent", methods=["GET"])
def memory_recent():
    """获取最近存储的长期记忆。"""
    if _memory_manager is None:
        return jsonify({"memories": []})
    n = int(request.args.get("limit", 10))
    items = _memory_manager._chroma.get_recent(limit=n)
    return jsonify({"memories": items})


@app.route("/api/memory/profile", methods=["GET"])
def memory_profile():
    """获取/更新用户画像。"""
    if _memory_manager is None:
        return jsonify({"profile": {}})
    return jsonify({"profile": _memory_manager.get_user_profile()})


@app.route("/")
def index():
    response = send_from_directory(app.static_folder, "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# ── 辅助 ────────────────────────────────────────────────

def _wav_to_pcm_chunks(wav_bytes: bytes, chunk_size: int = 4096) -> list[bytes]:
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
    return [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)]


def run_server(host: str = "0.0.0.0", port: int = 8081, debug: bool = False):
    logger.info("Voice Agent Web 服务启动: http://%s:%s", host, port)
    # threaded=True 是关键：Flask 默认单线程会卡住 SSE
    app.run(host=host, port=port, debug=debug, threaded=True)