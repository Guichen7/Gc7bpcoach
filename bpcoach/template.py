# -*- coding: utf-8 -*-
"""B/P 序列模板。

为什么把顺序抽成配置：KPL 的 18 手、排位征召的 16 手、以及不同赛年规则改动，
差别全在「谁在第几步、是禁还是选」。编码层不该把这些写死，
否则规则一变就要改代码。模板只描述**蓝红双方的动作序列**，不含英雄。
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from .errors import BPCoachError
from .heroes import config_dir

BAN = "ban"
PICK = "pick"


@dataclass(frozen=True)
class Step:
    index: int          # 从 0 开始
    side: int           # 1 = 蓝方，2 = 红方
    kind: str           # "ban" | "pick"

    @property
    def is_ban(self) -> bool:
        return self.kind == BAN

    @property
    def order_in_phase(self) -> str:
        """给前端用的简写：b1 / p2。"""
        return f"{'b' if self.is_ban else 'p'}{self.side}"

    def __str__(self) -> str:
        return self.order_in_phase


@dataclass(frozen=True)
class BPTemplate:
    id: str
    label: str
    steps: tuple[Step, ...]
    games: int = 1
    wins_needed: int = 1
    global_lock: str = "none"   # "none" | "picks" | "picks_and_bans"

    # ---------------- 构造 ----------------
    @classmethod
    def from_dict(cls, data: dict) -> "BPTemplate":
        steps = tuple(
            Step(index=i, side=int(s["side"]), kind=str(s["kind"]))
            for i, s in enumerate(data["steps"])
        )
        for s in steps:
            if s.side not in (1, 2):
                raise BPCoachError(f"模板 {data['id']} 第 {s.index + 1} 步 side 只能是 1 或 2")
            if s.kind not in (BAN, PICK):
                raise BPCoachError(f"模板 {data['id']} 第 {s.index + 1} 步 kind 只能是 ban/pick")

        tpl = cls(
            id=str(data["id"]),
            label=str(data.get("label", data["id"])),
            steps=steps,
            games=int(data.get("games", 1)),
            wins_needed=int(data.get("winsNeeded", (int(data.get("games", 1)) + 1) // 2)),
            global_lock=str(data.get("globalLock", "none")),
        )
        tpl.validate()
        return tpl

    @classmethod
    def load(cls, template_id: str, path: str | pathlib.Path | None = None) -> "BPTemplate":
        p = pathlib.Path(path) if path else config_dir() / "bp_templates.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        for item in data["templates"]:
            if item["id"] == template_id:
                return cls.from_dict(item)
        ids = ", ".join(t["id"] for t in data["templates"])
        raise BPCoachError(f"找不到模板 {template_id!r}，可用：{ids}")

    # ---------------- 校验 ----------------
    def validate(self) -> None:
        bans = {1: 0, 2: 0}
        picks = {1: 0, 2: 0}
        for s in self.steps:
            if s.is_ban:
                bans[s.side] += 1
            else:
                picks[s.side] += 1
        if picks[1] != picks[2]:
            raise BPCoachError(f"模板 {self.id} 双方选人数不等：{picks[1]} vs {picks[2]}")
        if picks[1] != 5:
            raise BPCoachError(f"模板 {self.id} 每方应有 5 手选人，实际 {picks[1]}")
        if bans[1] != bans[2]:
            raise BPCoachError(f"模板 {self.id} 双方 ban 数不等：{bans[1]} vs {bans[2]}")
        if self.global_lock not in ("none", "picks", "picks_and_bans"):
            raise BPCoachError(f"模板 {self.id} globalLock 取值非法：{self.global_lock}")
        if self.wins_needed * 2 - 1 != self.games:
            raise BPCoachError(
                f"模板 {self.id} 的 games={self.games} 与 winsNeeded={self.wins_needed} 不匹配"
            )

    # ---------------- 查询 ----------------
    def __len__(self) -> int:
        return len(self.steps)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def step(self, index: int) -> Step:
        if not 0 <= index < len(self.steps):
            raise BPCoachError(f"步数越界：{index}")
        return self.steps[index]

    def count(self, side: int, kind: str) -> int:
        return sum(1 for s in self.steps if s.side == side and s.kind == kind)

    def sequence_string(self) -> str:
        """紧凑表示，例如 b1-b2-b1-b2-p1-p2-..."""
        return "-".join(str(s) for s in self.steps)

    def __str__(self) -> str:
        return f"{self.id}（{len(self.steps)} 手）：{self.sequence_string()}"
