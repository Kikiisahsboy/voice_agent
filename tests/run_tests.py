# -*- coding: utf-8 -*-
"""轻量级测试 Runner — 在没有 pytest 的环境下用 stdlib 跑测试。

用法：
    python tests/run_tests.py
    python tests/run_tests.py tests/test_skill_manager.py
"""

import importlib
import importlib.util
import os
import sys
import traceback

# 强制 UTF-8 输出（Windows 默认 GBK 会让 ✓ ✗ 报错）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)              # voice_agent/
_PKG = os.path.join(_ROOT, "voice_agent")   # voice_agent/voice_agent/
for p in (_ROOT, _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)


def _discover_test_files(targets):
    if targets:
        return [t for t in targets if os.path.isfile(t)]
    out = []
    for fname in sorted(os.listdir(_HERE)):
        if fname.startswith("test_") and fname.endswith(".py"):
            out.append(os.path.join(_HERE, fname))
    return out


def _run_one_file(path):
    mod_name = "test_mod_" + os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fns = [
        (name, obj) for name, obj in vars(mod).items()
        if name.startswith("test_") and callable(obj)
    ]

    passed = []
    failed = []
    for name, fn in fns:
        try:
            fn()
            passed.append(name)
        except Exception as e:
            tb = traceback.format_exc()
            failed.append((name, str(e), tb))

    return passed, failed


def main():
    targets = sys.argv[1:]
    files = _discover_test_files(targets)
    if not files:
        print("未找到测试文件")
        sys.exit(1)

    total_pass = 0
    total_fail = 0

    OK = "[PASS]"
    NG = "[FAIL]"

    for f in files:
        print(f"\n--- {os.path.basename(f)} ---")
        passed, failed = _run_one_file(f)
        for name in passed:
            print(f"  {OK} {name}")
            total_pass += 1
        for name, err, tb in failed:
            print(f"  {NG} {name}: {err}")
            print(f"      {tb.splitlines()[-1] if tb else ''}")
            total_fail += 1

    print(f"\n{'='*60}")
    print(f"结果: {total_pass} 通过 / {total_fail} 失败 (共 {total_pass+total_fail})")
    print(f"{'='*60}")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()