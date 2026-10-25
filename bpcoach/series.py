# -*- coding: utf-8 -*-
"""系列赛上下文：处理「全局 B/P」。

这是 BPCoach 与普通阵容工具的分水岭：第 3 局能选什么，取决于第 1、2 局用掉了哪些英雄。
把跨局状态单独放一个对象，而不是塞进单局状态里，是为了：
  1) 单局编码保持「一局一状态」，便于 M4 做搜索和置换表；
  2) 全局锁定可以整体替换（不同赛制锁定规则不同）。

一、锁定的正确粒度（2026 规则原文决定）
腾讯官方《2026 年王者荣耀 KPL 年度总决赛赛事规则》6.2 全局 B/P：
    「己方使用过的英雄，不会出现在剩余比赛的己方阵容中，也不会出现在剩余比赛的
      对方禁用池中，即：每个英雄每一方只能使用一次。」
这句话有三层含义，缺一层就会把规则实现错：
  1. 锁是**按队伍**算的：蓝方第 1 局用了公孙离，红方第 2 局**仍然可以选**公孙离；
  2. 锁只作用于**选用**：己方用过的英雄，己方后续不能再选；
  3. ban 是「为对方设限」，所以只有当**对方**已经用过某英雄时，ban 它才等于空过。
早期版本用一条全局 `_used` 向量，会把红方第 2 局选公孙离误判为违规，故此处拆成两套。

二、为什么还要保留 `side=None` 的用法
复盘、脏数据校验、前端展示需要一个「保守视图」：任一方用过就标灰。
所以 lock_of / locked_ids / used_vector 都支持 side 省略，语义是「任一方用过」，
但**合法性判断一律显式传 side**，避免默认值悄悄改变规则。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .heroes import HeroPool

BLUE = 1
RED = 2


def other(side: int) -> int:
    """1 -> 2, 2 -> 1。"""
    if side not in (BLUE, RED):
        raise ValueError(f"side 只能是 1 或 2，收到 {side!r}")
    return RED if side == BLUE else BLUE


@dataclass
class GameRecord:
    """一局打完后的记录：只保留编码需要的英雄集合。"""

    game_index: int
    picks: dict[int, list[int]] = field(default_factory=dict)   # side -> 英雄 id
    bans: dict[int, list[int]] = field(default_factory=dict)    # side -> 英雄 id
    winner: int | None = None

    def all_picked(self) -> list[int]:
        return [h for hs in self.picks.values() for h in hs]

    def all_banned(self) -> list[int]:
        return [h for hs in self.bans.values() for h in hs]


class SeriesContext:
    """一个 BoN 系列赛的跨局状态。

    global_lock:
        "none"            不锁定（排位赛）
        "picks"           己方用过的英雄，己方后续不能再选（KPL，2026 规则）
        "picks_and_bans"  连己方 ban 过的一起锁（更严格的赛制）
    """

    def __init__(self, pool: HeroPool, global_lock: str = "picks") -> None:
        if global_lock not in ("none", "picks", "picks_and_bans"):
            raise ValueError(f"global_lock 非法：{global_lock}")
        self.pool = pool
        self.global_lock = global_lock
        self.games: list[GameRecord] = []
        # 按队伍分开的消耗表：_used[side][hero_id] ∈ {0,1}
        self._used: dict[int, list[int]] = {
            BLUE: [0] * pool.size,
            RED: [0] * pool.size,
        }

    # ---------------- 版本 / 比分 ----------------
    @property
    def game_index(self) -> int:
        """当前即将进行的是第几局（从 1 开始）。"""
        return len(self.games) + 1

    def score(self) -> dict[int, int]:
        s = {BLUE: 0, RED: 0}
        for g in self.games:
            if g.winner in s:
                s[g.winner] += 1
        return s

    # ---------------- 锁定 ----------------
    def lock_of(self, hero_id: int, side: int | None = None) -> bool:
        """该英雄是否已被「用掉」。

        side=None 为保守视图：任一方用过即 True（用于展示/复盘）。
        判合法性时必须传 side，语义是「这一方自己用过了」。
        """
        if self.global_lock == "none":
            return False
        if side is None:
            return self._used[BLUE][hero_id] == 1 or self._used[RED][hero_id] == 1
        return self._used[side][hero_id] == 1

    def locked_ids(self, side: int | None = None) -> list[int]:
        return [i for i in range(self.pool.size) if self.lock_of(i, side)]

    def used_vector(self, side: int | None = None) -> list[int]:
        """长度 N 的 0/1 向量：该英雄是否已被消耗（供模型做特征）。

        side=None 返回并集；传 side 返回该方自己的消耗表。
        """
        if side is not None:
            return list(self._used[side])
        return [
            1 if (self._used[BLUE][i] or self._used[RED][i]) else 0
            for i in range(self.pool.size)
        ]

    def ban_is_wasted(self, hero_id: int, banner_side: int) -> bool:
        """对 banner_side 来说，ban 这个英雄是不是等于空过。

        ban 的意义是「不让对方选到」，所以只有当对方已经用过它时才没意义。
        注意用「对方」而非「自己」：己方用过、对方没用的英雄，ban 掉依然有战术价值。
        """
        if self.global_lock == "none":
            return False
        return self._used[other(banner_side)][hero_id] == 1

    # ---------------- 记一局 ----------------
    def commit(self, record: GameRecord) -> None:
        if record.game_index != self.game_index:
            raise ValueError(
                f"局号不连续：当前应为第 {self.game_index} 局，收到第 {record.game_index} 局"
            )
        self.games.append(record)
        if self.global_lock == "none":
            return
        for side, heroes in record.picks.items():
            if side not in (BLUE, RED):
                continue
            for hid in heroes:
                self._used[side][hid] = 1
        if self.global_lock == "picks_and_bans":
            for side, heroes in record.bans.items():
                if side not in (BLUE, RED):
                    continue
                for hid in heroes:
                    self._used[side][hid] = 1

    def clone(self) -> "SeriesContext":
        new = SeriesContext(self.pool, self.global_lock)
        new._used = {BLUE: list(self._used[BLUE]), RED: list(self._used[RED])}
        new.games = [
            GameRecord(
                game_index=g.game_index,
                picks={k: list(v) for k, v in g.picks.items()},
                bans={k: list(v) for k, v in g.bans.items()},
                winner=g.winner,
            )
            for g in self.games
        ]
        return new

    def summary(self) -> str:
        s = self.score()
        if not self.games:
            return "系列赛尚未开始"
        side_name = {BLUE: "蓝方", RED: "红方"}
        lines = [f"已战 {len(self.games)} 局，比分 {s[BLUE]} : {s[RED]}"]
        for g in self.games:
            picks = " / ".join(
                f"{side_name.get(side, side)}[{'、'.join(self.pool.names(hs))}]"
                for side, hs in sorted(g.picks.items())
            )
            lines.append(f"  第 {g.game_index} 局：{picks}")
        lines.append(
            f"  已消耗（按队伍）：蓝方 {len(self.locked_ids(BLUE))} 个 / "
            f"红方 {len(self.locked_ids(RED))} 个"
        )
        return "\n".join(lines)
