@echo off
chcp 65001 >nul
title Voice Agent

echo ============================================
echo   Voice Agent - 语音聊天机器人
echo ============================================
echo.

:: 检查 Conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Conda，请先安装 Miniconda
    echo   下载: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: 激活环境
call conda activate voice_agent 2>nul
if %errorlevel% neq 0 (
    echo [提示] Conda 环境 voice_agent 不存在，尝试创建...
    echo   运行: conda env create -f voice_agent/environment.yml
    pause
    exit /b 1
)

:: 检查 Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] Ollama 服务未运行
    echo   请先启动 Ollama（双击桌面图标或在终端运行 ollama serve）
    echo.
    echo   如果尚未拉取模型，请运行:
    echo     ollama pull qwen2.5:1.5b
    echo.
    pause
    exit /b 1
)

:: 获取脚本所在目录
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

echo [启动] 正在启动 Voice Agent...
echo.

python voice_agent/main.py --config voice_agent/config.yaml

pause
