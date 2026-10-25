# -*- coding: utf-8 -*-
"""异常定义。所有面向使用者的报错都写成中文，方便直接回给教练/前端。"""

from __future__ import annotations


class BPCoachError(Exception):
    """本包所有业务异常的基类。"""


class UnknownHeroError(BPCoachError):
    """英雄名/别名无法在池中唯一解析。"""

    def __init__(self, raw: str, candidates: list[str] | None = None) -> None:
        msg = f"无法识别的英雄：{raw!r}"
        if candidates:
            msg += f"（是否想输入：{'、'.join(candidates)}）"
        super().__init__(msg)
        self.raw = raw
        self.candidates = candidates or []


class IllegalActionError(BPCoachError):
    """动作在当前局面下不合法（重复、已锁定、流程已结束等）。"""


class DraftFinishedError(BPCoachError):
    """B/P 已经走完，不能再执行动作。"""
