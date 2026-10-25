# -*- coding: utf-8 -*-
"""分路归属：处理「摇摆位」——同一英雄可在多条分路出场。

问题在哪
    之前 M2 的 `lane_fit` 是这么写的：把每个已选英雄的**所有**候选分路都标成「已占」。
    马超写着「对抗路 / 打野」，于是选一个马超就同时填掉了两条路，
    剩下的打野位被凭空补上，模型会误以为阵容已经齐了。
    同理，候选英雄有两条可打路时直接得 2 分，等于奖励「摇摆」本身，
    而真正该奖励的是「补上了还空着的那条路」。

怎么改
    把「英雄 → 分路」从 0/1 集合改成一个**概率分布** p(lane | hero)，
    再把「五个位置各一人」做成在 5×5 上的**单射分配**（一个英雄只能占一条路）：

        occupancy(l) = Σ_A  P(A) · 1[l ∈ A]      A 为把当前选人单射到分路的所有方案
        gap(l)       = 1 − occupancy(l)          这条路还空着的概率
        fill_value(h) = Σ_l p(l | h) · gap(l)    这一手补上真空位的期望收益 ∈ [0,1]

    单射枚举最多 5! = 120 种，可以精确算，不需要近似。
    fill_value 有两个好性质：摇摆英雄自动被折价（两条路各分走一部分概率），
    已经被队友大概率占掉的路也会自动降权 —— 这两点都是旧的 0/1 写法做不到的。

    概率从哪来：优先用 `config/lane_usage.json`，它是 tools/learn_lane_usage.py 从官方数据
    推出的分路分布 —— 拿「选手位置 × 英雄声明分路 × 召唤师技能（惩击=打野）」做 5×5 指派，
    比「按战报选用顺序猜分路」稳得多。校验：单分路英雄自洽率 99.9%，
    带惩击的选手 99.72% 被判到打野。
    没有观测数据时才退回人工声明的 `lanes` 字段（主路 0.75，其余均分）。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .heroes import CANONICAL_LANES

LANE_ORDER = CANONICAL_LANES   # 规范顺序在 heroes.py 定义，这里只做别名
# 人工声明的 lanes 字段退化成概率时的先验：主路 0.75，其余平分
DEFAULT_PRIMARY = 0.75
# 观测计数的平滑强度（相当于先验占几个样本）
ALPHA = 3.0
# 低于这个观测数就认为数据不足，退回声明值
MIN_OBS = 4
# 均匀底噪：给每条路留一点概率，理由有两个
#   ① 职业赛场确实出现非常规分路（赵怀真打辅助、司空震打中），概率不该是硬 0；
#   ② 纯硬 0 会让「两个英雄都只能打中路」这类阵容在单射分配时无解（概率全为 0），
#      底噪保证任何阵容都存在合法分配，代价只是极小的先验偏差。
EPS_FLOOR = 0.03


@dataclass(frozen=True)
class LaneModel:
    """英雄 → 分路概率分布 + 单射分配求解器。"""

    lanes: tuple[str, ...]
    dist: dict[int, tuple[float, ...]] = field(default_factory=dict)
    source: str = "declared"
    n_games: int = 0

    # ---------------- 构造 ----------------
    @classmethod
    def from_pool(cls, pool, usage: dict | None = None) -> "LaneModel":
        """usage: {"n_games": int, "heroes": {hero_name: {lane: count}}}（可缺省）。"""
        lanes = tuple(pool.lanes)
        counts = (usage or {}).get("heroes") or {}
        n_games = int((usage or {}).get("n_games", 0))
        dist: dict[int, tuple[float, ...]] = {}
        L = len(lanes)
        for hero in pool.heroes:
            base = cls._declared(hero, lanes)
            obs = counts.get(hero.name) or {}
            total = sum(obs.values())
            if total >= MIN_OBS:
                vec = [(obs.get(lane, 0.0) + ALPHA * base[i]) / (total + ALPHA)
                       for i, lane in enumerate(lanes)]
            else:
                vec = list(base)
            # 加底噪并归一化
            vec = [(1.0 - EPS_FLOOR) * v + EPS_FLOOR / L for v in vec]
            s = sum(vec)
            dist[hero.id] = tuple(v / s for v in vec)
        return cls(lanes=lanes, dist=dist,
                   source="usage+declared" if counts else "declared", n_games=n_games)

    @staticmethod
    def _declared(hero, lanes: tuple[str, ...]) -> tuple[float, ...]:
        playable = [l for l in (hero.lanes or (hero.lane,)) if l in lanes]
        if not playable:
            return tuple(1.0 / len(lanes) for _ in lanes)
        if len(playable) == 1:
            vec = [0.0] * len(lanes)
            vec[lanes.index(playable[0])] = 1.0
            return tuple(vec)
        others = (1.0 - DEFAULT_PRIMARY) / (len(playable) - 1)
        vec = []
        for lane in lanes:
            if lane == hero.lane and lane in playable:
                vec.append(DEFAULT_PRIMARY)
            elif lane in playable:
                vec.append(others)
            else:
                vec.append(0.0)
        return tuple(vec)

    # ---------------- 单英雄 ----------------
    def probs(self, hero_id: int) -> tuple[float, ...]:
        return self.dist.get(hero_id) or tuple(1.0 / len(self.lanes) for _ in self.lanes)

    def top_lanes(self, hero_id: int, k: int = 2) -> list[tuple[str, float]]:
        pairs = sorted(zip(self.lanes, self.probs(hero_id)), key=lambda kv: -kv[1])
        return [(l, p) for l, p in pairs[:k] if p > 0]

    def is_flex(self, hero_id: int, threshold: float = 0.15) -> bool:
        """摇摆位的判定：次选分路概率超过阈值。"""
        ps = sorted(self.probs(hero_id), reverse=True)
        return len(ps) > 1 and ps[1] >= threshold

    # ---------------- 阵容级 ----------------
    def _assignments(self, hero_ids: list[int]):
        """枚举把 hero_ids 单射到分路的所有方案，产出 (权重, 分路下标元组)。"""
        n = len(hero_ids)
        if n > len(self.lanes):
            hero_ids = hero_ids[: len(self.lanes)]
            n = len(hero_ids)
        for perm in itertools.permutations(range(len(self.lanes)), n):
            w = 1.0
            for hero, lane_idx in zip(hero_ids, perm):
                w *= self.probs(hero)[lane_idx]
                if w == 0.0:
                    break
            if w > 0.0:
                yield w, perm

    def occupancy(self, hero_ids: list[int]) -> list[float]:
        """每条分路被当前选人占用的期望（0..1）。"""
        L = len(self.lanes)
        acc = [0.0] * L
        z = 0.0
        for w, perm in self._assignments(list(hero_ids)):
            z += w
            for lane_idx in perm:
                acc[lane_idx] += w
        if z <= 0:
            return [0.0] * L
        return [a / z for a in acc]

    def gap(self, hero_ids: list[int]) -> list[float]:
        """每条路还空着的概率。"""
        return [1.0 - o for o in self.occupancy(list(hero_ids))]

    def fill_value(self, picked: list[int], hero_id: int) -> float:
        """这一手能补上「真空位」的期望概率。0 表示位置已齐或纯挤位。"""
        gaps = self.gap(picked)
        return sum(p * g for p, g in zip(self.probs(hero_id), gaps))

    def log_assignment(self, hero_ids: list[int]) -> float:
        """最优分配的对数似然。M3 用它衡量「这五个人能不能排成合理阵容」。"""
        best = 0.0
        for w, _ in self._assignments(list(hero_ids)):
            if w > best:
                best = w
        return math.log(best) if best > 0 else float("-inf")

    def resolve(self, hero_ids: list[int]) -> list[tuple[int, str, float]]:
        """最可能的分配方案：[(hero_id, lane, prob)]，供前端展示与复盘。"""
        best: tuple[float, tuple[int, ...]] | None = None
        for w, perm in self._assignments(list(hero_ids)):
            if best is None or w > best[0]:
                best = (w, perm)
        if best is None:
            return []
        return [(h, self.lanes[i], self.probs(h)[i]) for h, i in zip(hero_ids, best[1])]

    def to_dict(self) -> dict:
        return {
            "lanes": list(self.lanes),
            "source": self.source,
            "n_games": self.n_games,
            "dist": {str(h): [round(x, 6) for x in v] for h, v in self.dist.items()},
        }
