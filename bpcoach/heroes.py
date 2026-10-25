# -*- coding: utf-8 -*-
"""英雄池：名称 ↔ 下标 的双向映射，是全系统唯一的 ID 真源。

关键约定：**英雄下标（hero id）一旦发布就不许变**。
后面 M2 的转移矩阵、M3 的模型权重、M4 的置换表都是按这个下标对齐的，
中途插入一个英雄会让所有矩阵错位。新增英雄只能追加到池尾，并同步升 patch 号。
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field

from .errors import BPCoachError, UnknownHeroError

DEFAULT_CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "config"

# 分路的**规范顺序**。必须写死，不能从英雄池里「首次出现」的顺序推：
# 一旦有人调整某个英雄的分路，推导顺序就会变，而分路向量的每一列会跟着错位，
# 所有已训练的模型和特征都会对不上号。
CANONICAL_LANES = ("对抗路", "打野", "中路", "发育路", "游走")


def config_dir() -> pathlib.Path:
    """配置目录，可用环境变量 BPCOACH_CONFIG 覆盖（便于换成别的赛季/别的游戏）。"""
    env = os.environ.get("BPCOACH_CONFIG")
    return pathlib.Path(env) if env else DEFAULT_CONFIG_DIR


@dataclass(frozen=True)
class Hero:
    id: int
    name: str
    lane: str
    lanes: tuple[str, ...] = field(default_factory=tuple)
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:  # 方便直接打印
        return self.name


class HeroPool:
    """英雄池。构造后不可变，可安全在线程/进程间共享。"""

    def __init__(
        self,
        heroes: list[Hero],
        season: str = "",
        codename: str = "",
        patch: str = "",
        start_date: str = "",
        note: str = "",
        pick_groups: list[list[int]] | None = None,
    ) -> None:
        self._heroes = tuple(heroes)
        self.season = season
        self.codename = codename
        self.patch = patch
        self.start_date = start_date
        self.note = note

        self._by_id: dict[int, Hero] = {}
        self._by_name: dict[str, int] = {}
        for h in self._heroes:
            if h.id in self._by_id:
                raise BPCoachError(f"英雄 id 重复：{h.id}")
            if h.name in self._by_name:
                raise BPCoachError(f"英雄名重复：{h.name}")
            self._by_id[h.id] = h
            self._by_name[h.name] = h.id
            for alias in h.aliases:
                if alias in self._by_name and self._by_name[alias] != h.id:
                    raise BPCoachError(f"别名冲突：{alias!r} 同时指向 {h.name} 与其他英雄")
                self._by_name[alias] = h.id

        # 校验 id 必须是 0..N-1 的连续整数，保证「下标即 id」
        ids = sorted(self._by_id)
        if ids != list(range(len(self._heroes))):
            raise BPCoachError("英雄 id 必须是从 0 开始的连续整数")

        self._lane_usage: dict | None = None
        self._lane_model = None

        # 选用互斥组：元流之子的 5 张职业卡共用 1 个选用名额。
        # 只影响「选」，不影响「禁」——ban 掉辅助，对面仍可选坦克（官方数据实证）。
        self._groups: tuple[tuple[int, ...], ...] = tuple(
            tuple(sorted(g)) for g in (pick_groups or []) if len(g) > 1
        )
        self._group_of: dict[int, int] = {}
        for gi, members in enumerate(self._groups):
            for hid in members:
                if hid not in self._by_id:
                    raise BPCoachError(f"选用互斥组含未知英雄 id={hid}")
                if hid in self._group_of and self._group_of[hid] != gi:
                    raise BPCoachError(f"英雄 id={hid} 同时出现在两个选用互斥组")
                self._group_of[hid] = gi

    # ---------------- 构造 ----------------
    @classmethod
    def load(cls, path: str | pathlib.Path | None = None) -> "HeroPool":
        p = pathlib.Path(path) if path else config_dir() / "heroes_s45.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "HeroPool":
        """从已解析的 JSON dict 构建（bundle 加载用，与 load 同一份校验）。"""
        heroes = [
            Hero(
                id=int(h["id"]),
                name=h["name"],
                lane=h.get("lane", ""),
                lanes=tuple(h.get("lanes", [])),
                aliases=tuple(h.get("aliases", [])),
            )
            for h in data["heroes"]
        ]
        return cls(
            heroes,
            season=data.get("season", ""),
            codename=data.get("codename", ""),
            patch=data.get("patch", ""),
            start_date=data.get("startDate", ""),
            note=data.get("note", ""),
            pick_groups=[g["members"] for g in data.get("pickGroups", [])],
        )

    # ---------------- 基本属性 ----------------
    def __len__(self) -> int:
        return len(self._heroes)

    @property
    def size(self) -> int:
        return len(self._heroes)

    @property
    def heroes(self) -> tuple[Hero, ...]:
        return self._heroes

    @property
    def lanes(self) -> tuple[str, ...]:
        """规范顺序的分路列表（见 CANONICAL_LANES）；池里出现的额外分路排在后面。"""
        present = {h.lane for h in self._heroes if h.lane}
        out = [l for l in CANONICAL_LANES if l in present]
        for h in self._heroes:
            if h.lane and h.lane not in out:
                out.append(h.lane)
        return tuple(out)

    def __iter__(self):
        return iter(self._heroes)

    # ---------------- 解析 ----------------
    def hero(self, hero_id: int) -> Hero:
        try:
            return self._by_id[hero_id]
        except KeyError:
            raise UnknownHeroError(f"id={hero_id}") from None

    def index(self, raw: str | int | Hero) -> int:
        """把名称/别名/id 统一解析成下标。"""
        if isinstance(raw, Hero):
            return raw.id
        if isinstance(raw, int):
            if raw not in self._by_id:
                raise UnknownHeroError(f"id={raw}")
            return raw
        key = str(raw).strip()
        if key in self._by_name:
            return self._by_name[key]
        # 容错：去空格、去「英雄」后缀、忽略大小写
        key2 = key.replace(" ", "").replace("　", "")
        if key2 in self._by_name:
            return self._by_name[key2]
        candidates = [h.name for h in self._heroes if key and key in h.name][:5]
        raise UnknownHeroError(key, candidates)

    def name_of(self, hero_id: int) -> str:
        return self.hero(hero_id).name

    def names(self, hero_ids) -> list[str]:
        return [self.name_of(i) for i in hero_ids]

    def lane_of(self, hero_id: int) -> str:
        return self.hero(hero_id).lane

    # ---------------- 选用互斥组 ----------------
    @property
    def pick_groups(self) -> tuple[tuple[int, ...], ...]:
        return self._groups

    def pick_group(self, hero_id: int) -> tuple[int, ...]:
        """该英雄所属的选用互斥组（含自身）；无组时返回 (hero_id,)。"""
        gi = self._group_of.get(hero_id)
        return self._groups[gi] if gi is not None else (hero_id,)

    def pick_excludes(self, hero_id: int, picked: set[int] | list[int]) -> int | None:
        """若已选集合里已有同组英雄，返回冲突的那个 id，否则 None。

        给状态层用：选人前问一句「这手选下去会不会跟场上撞组」。
        """
        for other in self.pick_group(hero_id):
            if other != hero_id and other in picked:
                return other
        return None

    # ---------------- 预计算缓存（供下游频繁调用） ----------------
    def one_hot(self, hero_ids) -> list[int]:
        """给定英雄集合，返回长度 N 的 0/1 向量。"""
        vec = [0] * self.size
        for i in hero_ids:
            vec[i] = 1
        return vec

    def lane_matrix(self) -> list[list[int]]:
        """N × L 的 0/1 矩阵：英雄在哪条分路可打。

        注意这是「能不能打」的硬集合；做阵容分路推断请用 lane_model()，
        后者给出概率分布并能处理摇摆位（同一英雄两条路按比例分配）。
        """
        lanes = self.lanes
        mat = [[0] * len(lanes) for _ in range(self.size)]
        for h in self._heroes:
            for lane in h.lanes or (h.lane,):
                if lane in lanes:
                    mat[h.id][lanes.index(lane)] = 1
        return mat

    # ---------------- 分路模型 ----------------
    def lane_usage(self) -> dict:
        """读 config/lane_usage.json（由 tools/learn_lane_usage.py 生成），缺省为空。"""
        if self._lane_usage is None:
            path = config_dir() / "lane_usage.json"
            self._lane_usage = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            )
        return self._lane_usage

    def lane_model(self):
        """英雄 → 分路概率分布（含摇摆位）。首次调用时构建，之后缓存。"""
        if self._lane_model is None:
            from .lane import LaneModel

            self._lane_model = LaneModel.from_pool(self, self.lane_usage())
        return self._lane_model

    def attach_lane_usage(self, usage: dict) -> None:
        """显式注入分路统计（bundle 加载用），并使已缓存的 lane_model 失效。"""
        self._lane_usage = usage
        self._lane_model = None

    def __repr__(self) -> str:
        return f"<HeroPool {self.season} patch={self.patch} size={self.size}>"
