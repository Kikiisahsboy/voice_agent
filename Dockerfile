# Voice Agent — Production Dockerfile
# 默认 headless：不安装 PyAudio（本地不需要服务端喇叭），但保留容器内的 TTS 字节合成能力。
# 如需服务端本地播放，在构建时取消注释 portaudio + pyaudio 部分。

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 系统依赖：chroma 需要 sqlite3 + build tools（sentence-transformers 需要编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 如需服务端播放：取消下面注释
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         portaudio19-dev \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 如果要服务端 TTS 播放：pip install pyaudio
# RUN pip install pyaudio

COPY voice_agent/ ./voice_agent/
COPY tests/ ./tests/
COPY eval/ ./eval/

# ChromaDB 数据持久化
RUN mkdir -p /app/voice_agent/chroma_data
VOLUME ["/app/voice_agent/chroma_data"]

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8081/api/health || exit 1

CMD ["python", "voice_agent/main.py", "--host", "0.0.0.0", "--port", "8081"]