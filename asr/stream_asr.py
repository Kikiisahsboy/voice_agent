# -*- coding: utf-8 -*-
"""流式 ASR 服务 — Vosk 包装，支持 partial/final 回调。"""

import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class StreamASR:
    """流式 Vosk ASR，薄包装 LocalASREngine 逻辑。"""

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
    ):
        import vosk

        logger.info("加载 Vosk 模型: %s", model_path)
        self._model = vosk.Model(model_path)
        self._recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
        self._recognizer.SetWords(False)
        self._sample_rate = sample_rate
        self._on_partial = on_partial
        self._on_final = on_final
        logger.info("StreamASR 就绪")

    def feed(self, data: bytes) -> Optional[dict]:
        """
        输入 PCM 音频块，返回识别结果。

        Returns:
            {'type': 'partial', 'text': str} | {'type': 'final', 'text': str} | None
        """
        if self._recognizer.AcceptWaveform(data):
            result = json.loads(self._recognizer.Result())
            text = result.get("text", "").strip().replace(" ", "")
            if text:
                if self._on_final:
                    self._on_final(text)
                return {"type": "final", "text": text}
        else:
            partial = json.loads(self._recognizer.PartialResult())
            text = partial.get("partial", "").strip()
            if text:
                if self._on_partial:
                    self._on_partial(text)
                return {"type": "partial", "text": text}
        return None

    def finalize(self) -> Optional[dict]:
        """获取最终结果（语音结束后调用）。"""
        result = json.loads(self._recognizer.FinalResult())
        text = result.get("text", "").strip().replace(" ", "")
        if text:
            if self._on_final:
                self._on_final(text)
            return {"type": "final", "text": text}
        return None

    def reset(self):
        """重置识别器（新一轮对话前调用）。"""
        self._recognizer.Reset()
