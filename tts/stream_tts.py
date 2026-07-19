# -*- coding: utf-8 -*-
"""流式 TTS 服务 — edge-tts 流式合成 + miniaudio 解码 + PyAudio 播放。"""

import asyncio
import logging
import queue
import threading
from typing import Callable, Generator, Optional

logger = logging.getLogger(__name__)

try:
    import pyaudio
    _PYAUDIO_OK = True
except ImportError:
    pyaudio = None
    _PYAUDIO_OK = False

import edge_tts
import miniaudio


class StreamTTS:
    """流式 TTS：边合成边播放，支持随时打断。"""

    VOICE = "zh-CN-XiaoxiaoNeural"
    STREAM_CHUNK = 1024

    def __init__(self, sample_rate: int = 24000):
        if not _PYAUDIO_OK:
            raise ImportError("需要 pyaudio: pip install pyaudio")

        self.sample_rate = sample_rate
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stop_event = threading.Event()
        self._playing = False

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _stream_tts(self, text: str, pcm_queue: queue.Queue):
        communicate = edge_tts.Communicate(text, self.VOICE)
        mp3_buffer = bytearray()

        async for chunk in communicate.stream():
            if self._stop_event.is_set():
                break
            if chunk["type"] == "audio":
                mp3_buffer.extend(chunk["data"])
                if len(mp3_buffer) >= 4096:
                    pcm = self._decode_mp3(bytes(mp3_buffer))
                    if pcm:
                        pcm_queue.put(pcm)
                    mp3_buffer.clear()

        if mp3_buffer and not self._stop_event.is_set():
            pcm = self._decode_mp3(bytes(mp3_buffer))
            if pcm:
                pcm_queue.put(pcm)

        pcm_queue.put(None)

    def _decode_mp3(self, data: bytes) -> Optional[bytes]:
        try:
            decoded = miniaudio.decode(
                data,
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=1,
                sample_rate=self.sample_rate,
            )
            return decoded.samples.tobytes()
        except Exception as e:
            logger.debug("MP3 解码失败: %s", e)
            return None

    def speak(self, text: str) -> bool:
        """流式合成并播放。返回 True=完整播放, False=被打断。"""
        if not text.strip():
            return True

        logger.info("TTS: %s", text[:60])
        self._stop_event.clear()
        self._playing = True

        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

        stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.STREAM_CHUNK,
        )

        pcm_queue: queue.Queue = queue.Queue()
        future = asyncio.run_coroutine_threadsafe(
            self._stream_tts(text, pcm_queue), self._loop
        )

        interrupted = False
        try:
            while True:
                if self._stop_event.is_set():
                    interrupted = True
                    break
                try:
                    pcm = pcm_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if pcm is None:
                    break
                stream.write(pcm)
        finally:
            stream.stop_stream()
            stream.close()
            future.cancel()
            self._playing = False

        if interrupted:
            logger.info("TTS 被打断")
            return False
        logger.info("TTS 播放完毕")
        return True

    def speak_stream(
        self,
        text_generator: Generator[str, None, None],
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
    ) -> bool:
        """从 LLM 文本流边收边合成并播放。"""
        buffer = ""
        for chunk in text_generator:
            buffer += chunk
            if len(buffer) >= 20 and any(d in buffer for d in "。！？\n"):
                self.speak(buffer)
                buffer = ""
        if buffer.strip():
            self.speak(buffer)
        return True

    def stop(self):
        if self._playing:
            self._stop_event.set()
            logger.info("TTS 打断信号已发送")

    def close(self):
        self.stop()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
