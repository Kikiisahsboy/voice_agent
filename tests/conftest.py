"""共享 fixture：把 voice_agent/ 加入 sys.path。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(_ROOT, "voice_agent")
for p in (_ROOT, _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)