# -*- coding: utf-8 -*-
"""计算器 Skill。"""

import ast
import operator
from typing import Any

from voice_agent.agent.skill_manager import Skill

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(expression: str) -> Any:
    """安全计算数学表达式，仅允许基本运算和数字。"""

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"不支持的运算符: {op_type}")
            return _SAFE_OPS[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"不支持的运算符: {op_type}")
            return _SAFE_OPS[op_type](_eval(node.operand))
        if isinstance(node, ast.Constant):
            return node.value
        raise ValueError(f"不支持的表达式类型: {type(node)}")

    tree = ast.parse(expression.strip(), mode="eval")
    return _eval(tree)


def _calculate(expression: str) -> str:
    try:
        result = _safe_eval(expression)
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return str(result)
    except Exception as e:
        return f"计算出错: {e}"


skill_calculator = Skill(
    name="calculate",
    description="执行数学计算。当用户要求计算、求和、求积等数学运算时使用。如'1+2*3'、'100除以5'。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，仅含数字和基本运算符，如 '3+5*2'",
            }
        },
        "required": ["expression"],
    },
    handler=_calculate,
    category="utility",
)
